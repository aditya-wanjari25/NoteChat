import streamlit as st
from dotenv import load_dotenv
import os
import tempfile
from langchain.chat_models import init_chat_model
from langchain_openai import OpenAIEmbeddings
from langchain_pinecone import PineconeVectorStore
from pinecone import Pinecone, ServerlessSpec
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.tools import tool
from langgraph.prebuilt import create_react_agent
from langgraph.checkpoint.memory import MemorySaver
from typing import List

# For Word document processing
from docx import Document

# Load environment variables
load_dotenv()
LANGSMITH_API_KEY = os.environ["LANGSMITH"]
OPENAI_API_KEY = os.environ["OPENAI_KEY"]
PINECONE_API_KEY = os.environ["PINECONE_API_KEY"]

def create_pinecone_index(index_name, dimension):
    metric = "cosine"
    pc = Pinecone(api_key=PINECONE_API_KEY)
    pc.create_index(
        name=index_name,
        dimension=dimension,
        metric=metric,
        spec=ServerlessSpec(cloud="aws", region="us-east-1")
    )
    return pc.Index(index_name)

def process_files(files):
    documents = []
    for file in files:
        filename = file.name.lower()
        if filename.endswith('.pdf'):
            # Write the uploaded PDF file to a temporary file
            with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
                tmp.write(file.read())
                tmp_path = tmp.name
            loader = PyPDFLoader(tmp_path)
            documents.extend(loader.load())
            # Optionally, remove the temporary file after processing:
            os.remove(tmp_path)
        elif filename.endswith(('.docx', '.doc')):
            # For Word documents, Document can process file-like objects directly.
            doc = Document(file)
            text = "\n".join([para.text for para in doc.paragraphs])
            # Create a simple object similar to langchain's Document object
            DocumentObj = type("Document", (object,), {}) 
            doc_obj = DocumentObj()
            setattr(doc_obj, "page_content", text)
            setattr(doc_obj, "metadata", {"source": file.name})
            documents.append(doc_obj)
        else:
            st.warning(f"Unsupported file type: {file.name}. Only PDF and Word files are supported.")
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        add_start_index=True,
    )
    all_splits = text_splitter.split_documents(documents)
    return all_splits

def add_docs_vectordb(files: List, vector_store):
    all_splits = process_files(files)
    document_ids = vector_store.add_documents(documents=all_splits)
    return document_ids

@tool(response_format="content_and_artifact")
def retrieve(query: str):
    """Retrieve information related to a query."""
    retrieved_docs = vector_store.similarity_search(query, k=3)
    serialized = "\n\n".join(
        (f"Source: {doc.metadata}\nContent: {doc.page_content}")
        for doc in retrieved_docs
    )
    return serialized, retrieved_docs

def stream_ai_response(input_message, graph, config):
    for step in graph.stream(
        {"messages": [{"role": "user", "content": input_message}]},
        stream_mode="values",
        config=config,
    ):
        step["messages"][-1].pretty_print()

def get_ai_response(input_message, graph, config):
    result = graph.invoke(
        {"messages": [{"role": "user", "content": input_message}]},
        config=config
    )
    return result["messages"][-1].content

def delete_vectors():
    index = Pinecone(PINECONE_API_KEY).Index("myindex")
    return index.delete(delete_all=True)

# Initialize language model, embeddings, vector store, and agent
llm = init_chat_model("gpt-4o-mini", model_provider="openai", api_key=OPENAI_API_KEY)
embeddings_dimension = 3072
embeddings = OpenAIEmbeddings(model="text-embedding-3-large", api_key=OPENAI_API_KEY, dimensions=embeddings_dimension)

# Assume the index already exists. Otherwise, create it:
# index = create_pinecone_index("myindex", embeddings_dimension)
index = Pinecone(PINECONE_API_KEY).Index("myindex")
vector_store = PineconeVectorStore(embedding=embeddings, index=index)

memory = MemorySaver()
agent_executor = create_react_agent(llm, [retrieve], checkpointer=memory)
config = {"configurable": {"thread_id": "1a"}}

st.set_page_config(page_title="Custom RAG", page_icon="📝")

# Sidebar for document actions
with st.sidebar:
    st.header("Actions")
    uploaded_files = st.file_uploader("Upload Document", type=["pdf", "docx", "doc"], accept_multiple_files=True)
    if st.button("Add Documents to VectorDB"):
        if uploaded_files:
            add_docs_vectordb(uploaded_files, vector_store)
            st.success("Documents added to the vector store!")
        else:
            st.warning("Please upload a document first.")
    
    if st.button("Delete Vectors"):
        try:
            delete_vectors()
            st.success("Vectors deleted!")
        except:
            st.warning("Vector store remains empty.")

st.markdown(
    """
    <style>
    .chat-bubble {
        padding: 1rem;
        border-radius: 10px;
        margin-bottom: 1rem;
        max-width: 80%;
        box-shadow: 0px 4px 8px rgba(0,0,0,0.1);
        display: flex;
        align-items: center;
    }
    .chat-bubble.user {
        background-color: #007BFF;
        color: #ffffff;
        justify-content: flex-end;
        text-align: right;
        margin-left: auto;
    }
    .chat-bubble.assistant {
        background-color: #28a745;
        color: #ffffff;
        justify-content: flex-start;
        text-align: left;
        margin-right: auto;
    }
    .icon {
        font-size: 1.5rem;
        margin-right: 0.5rem;
    }
    </style>
    """,
    unsafe_allow_html=True
)

st.title("Custom RAG! Chat with Notes!")

if "messages" not in st.session_state:
    st.session_state.messages = [{
        "role": "assistant",
        "content": "Hey There! I'm here to assist you answer your questions."
    }]

for message in st.session_state.messages:
    if message["role"] == "user":
        st.markdown(
            f"<div class='chat-bubble user'><span class='icon'>👤</span><div>{message['content']}</div></div>", 
            unsafe_allow_html=True
        )
    else:
        st.markdown(
            f"<div class='chat-bubble assistant'><span class='icon'>🤖</span><div>{message['content']}</div></div>", 
            unsafe_allow_html=True
        )

prompt = st.chat_input("Ask me anything about your document...")

if prompt:
    st.markdown(
        f"<div class='chat-bubble user'><span class='icon'>👤</span><div>{prompt}</div></div>", 
        unsafe_allow_html=True
    )
    st.session_state.messages.append({
        "role": "user",
        "content": prompt
    })
    with st.spinner("Thinking..."):
        response = get_ai_response(prompt, agent_executor, config)
    st.markdown(
        f"<div class='chat-bubble assistant'><span class='icon'>🤖</span><div>{response}</div></div>", 
        unsafe_allow_html=True
    )
    st.session_state.messages.append({
        "role": "assistant",
        "content": response
    })
