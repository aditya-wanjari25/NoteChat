import streamlit as st

# Set page configuration as the very first command
st.set_page_config(page_title="NoteChat", page_icon="📝")

import os
import boto3
from dotenv import load_dotenv
import io
from langchain.document_loaders import PyPDFLoader, PyPDFium2Loader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from pinecone import Pinecone, ServerlessSpec
from langchain_pinecone import PineconeVectorStore
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain_openai import ChatOpenAI
from langgraph.checkpoint.memory import MemorySaver
import random

load_dotenv()

LOCAL_PATH = "./uploaded_files"
LOCAL_DOWNLOAD_PATH = "./downloaded_files"

# AWS S3 Configuration
AWS_ACCESS_KEY = os.getenv("AWS_ACCESS_KEY")
AWS_SECRET_KEY = os.getenv("AWS_SECRET_KEY")
AWS_REGION = os.getenv("AWS_REGION")
S3_BUCKET_NAME = os.getenv("S3_BUCKET_NAME")
OPENAI_API_KEY = os.environ["OPENAI_KEY"]

# Initialize S3 client
s3 = boto3.client(
    "s3",
    aws_access_key_id=AWS_ACCESS_KEY,
    aws_secret_access_key=AWS_SECRET_KEY,
    region_name=AWS_REGION,
)

pc = Pinecone(api_key=os.environ["PINECONE_API_KEY"])
embeddings = OpenAIEmbeddings(api_key=os.environ["OPENAI_KEY"])
llm = ChatOpenAI(model="gpt-4o-mini", api_key=OPENAI_API_KEY)


def upload_file(file_path):
    # File uploader
    uploaded_file = st.file_uploader("Choose a file", type=["txt", "docx", "pdf"])
    if uploaded_file is not None:
        save_path = os.path.join(file_path, uploaded_file.name)
        with open(save_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
        # Removed green success message


def upload_to_s3():
    uploaded_file = st.file_uploader("Choose a file", type=["txt", "docx", "pdf"])
    if uploaded_file is not None:
        file_name = uploaded_file.name
        # Upload file to S3 (notification removed)
        try:
            s3.upload_fileobj(uploaded_file, S3_BUCKET_NAME, file_name)
        except Exception as e:
            st.error(f"❌ Failed to upload file: {e}")
        return file_name


def download_s3_file(file_name):
    if not file_name:
        st.error("❌ No file name provided")
        return None
    try:
        os.makedirs(LOCAL_DOWNLOAD_PATH, exist_ok=True)
        local_file_path = os.path.join(LOCAL_DOWNLOAD_PATH, file_name)
        s3.download_file(S3_BUCKET_NAME, file_name, local_file_path)
        # Removed processing notification
        return local_file_path
    except Exception as e:
        st.error(f"❌ Failed to download file: {e}")
        return None


def document_loader_and_chunking(local_file_path):
    try:
        loader = PyPDFLoader(local_file_path)
        documents = loader.load()
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=100)
        all_splits = text_splitter.split_documents(documents)
        return all_splits
    except Exception as e:
        st.error(f"❌ Failed to load and chunk file: {e}")
        return None


def upsert_documents(documents):
    index_name = "langchainvectordb"
    index = pc.Index(index_name)
    vector_store = PineconeVectorStore(index=index, embedding=embeddings)
    vector_store.add_documents(documents)
    return vector_store


# Retrieval
from langgraph.graph import MessagesState, StateGraph
from langchain_core.tools import tool

graph_builder = StateGraph(MessagesState)

@tool(response_format="content_and_artifact")
def retrieve(query: str):
    """Retrieve information related to a query."""
    retrieved_docs = vector_store.similarity_search(query, k=2)
    serialized = "\n\n".join(
        (f"Source: {doc.metadata}\nContent: {doc.page_content}")
        for doc in retrieved_docs
    )
    return serialized, retrieved_docs


from langchain_core.messages import SystemMessage
from langgraph.prebuilt import ToolNode

def query_or_respond(state: MessagesState):
    """Generate tool call for retrieval or respond."""
    llm_with_tools = llm.bind_tools([retrieve])
    response = llm_with_tools.invoke(state["messages"])
    return {"messages": [response]}


tools = ToolNode([retrieve])


def generate(state: MessagesState):
    """Generate answer."""
    recent_tool_messages = []
    for message in reversed(state["messages"]):
        if message.type == "tool":
            recent_tool_messages.append(message)
        else:
            break
    tool_messages = recent_tool_messages[::-1]
    docs_content = "\n\n".join(doc.content for doc in tool_messages)
    system_message_content = (
        "You are an assistant for question-answering tasks. "
        "Use the following pieces of retrieved context to answer the question. "
        "These pieces are from a user's notes so they might be messy. If you don't know the answer, say so. "
        "Use three sentences maximum and keep the answer concise."
        "\n\n" +
        docs_content
    )
    conversation_messages = [
        message
        for message in state["messages"]
        if message.type in ("human", "system")
        or (message.type == "ai" and not message.tool_calls)
    ]
    prompt = [SystemMessage(system_message_content)] + conversation_messages
    response = llm.invoke(prompt)
    return {"messages": [response]}


from langgraph.graph import END
from langgraph.prebuilt import ToolNode, tools_condition
from langgraph.checkpoint.memory import MemorySaver

def setup_graph():
    graph_builder.add_node(query_or_respond)
    graph_builder.add_node(tools)
    graph_builder.add_node(generate)
    graph_builder.set_entry_point("query_or_respond")
    graph_builder.add_conditional_edges(
        "query_or_respond",
        tools_condition,
        {END: END, "tools": "tools"},
    )
    graph_builder.add_edge("tools", "generate")
    graph_builder.add_edge("generate", END)
    config = {"configurable": {"thread_id": "1"}}
    memory = MemorySaver()
    graph = graph_builder.compile(checkpointer=memory)
    return graph, config


def run_graph(input_message, graph, config):
    for step in graph.stream(
        {"messages": [{"role": "user", "content": input_message}]},
        stream_mode="values",
        config=config,
    ):
        message = step["messages"][-1]
    return message.content


# -----------------------------------------------------------------------------
# Custom CSS for styling (enhanced colors, modern look, and icons)
# -----------------------------------------------------------------------------
st.markdown(
    """
    <style>
    /* Overall page styling */
    body {
        background-color: #f0f4f8;
        color: #f5f5f5;
    }
    /* Container styling */

    /* Title styling */
    .app-title {
        font-family: 'Segoe UI', sans-serif;
        font-weight: 600;
        font-size: 2.5rem;
        color: #f0f0f0;
        text-align: center;
        margin-bottom: 1rem;
    }
    /* Chat bubble styling */
    .chat-bubble {
        padding: 1rem;
        border-radius: 10px;
        margin-bottom: 1rem;
        box-shadow: 0px 4px 8px rgba(0,0,0,0.2);
        font-family: 'Segoe UI', sans-serif;
        line-height: 1.4;
        display: flex;
        align-items: flex-start;
    }
    .chat-bubble .icon {
        font-size: 1.5rem;
        margin-right: 0.5rem;
    }
    .chat-bubble.user {
        background-color: #007BFF;
        color: #ffffff;
        justify-content: flex-end;
    }
    .chat-bubble.assistant {
        background-color: #28a745;
        color: #ffffff;
        justify-content: flex-start;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# -----------------------------------------------------------------------------
# Sidebar Instructions
# -----------------------------------------------------------------------------
st.sidebar.markdown("### NoteChat Instructions")
st.sidebar.info(
    "1. Upload your document to start chatting.\n"
    "2. Ask any questions related to your document in the chat input below.\n"
    "3. Enjoy your conversation with our assistant!"
)

# -----------------------------------------------------------------------------
# Main container for the app
# -----------------------------------------------------------------------------
st.markdown('<div class="main-container">', unsafe_allow_html=True)

st.markdown("<div class='app-title'>Welcome to NoteChat!</div>", unsafe_allow_html=True)
st.write("This is a simple RAG app designed to help you interact with your document.")

# -----------------------------------------------------------------------------
# Initialize or load graph, configuration, and session state
# -----------------------------------------------------------------------------
graph, config = setup_graph()

if "messages" not in st.session_state:
    st.session_state.messages = []

# -----------------------------------------------------------------------------
# File upload and document processing
# -----------------------------------------------------------------------------
file_name = upload_to_s3()

if file_name:
    local_file_path = download_s3_file(file_name)
    all_splits = document_loader_and_chunking(local_file_path)
    vector_store = upsert_documents(all_splits)
    
    # Notify the user that their document is ready to chat
    st.info("Your document is ready to chat!")

    # -----------------------------------------------------------------------------
    # Display chat history with styled chat bubbles and icons
    # -----------------------------------------------------------------------------
    for message in st.session_state.messages:
        role = message["role"]
        content = message["content"]
        if role == "user":
            icon = "👤"
        else:
            icon = "🤖"
        bubble_class = "chat-bubble " + role
        st.markdown(
            f"<div class='{bubble_class}'><span class='icon'>{icon}</span><div>{content}</div></div>", 
            unsafe_allow_html=True
        )

    # -----------------------------------------------------------------------------
    # Accept and process user input
    # -----------------------------------------------------------------------------
    prompt = st.chat_input("Ask me anything about your document...")
    if prompt:
        # Display user message with icon
        st.markdown(
            f"<div class='chat-bubble user'><span class='icon'>👤</span><div>{prompt}</div></div>", 
            unsafe_allow_html=True
        )
        st.session_state.messages.append({
            "role": "user",
            "content": prompt
        })
        with st.spinner("Thinking..."):
            response = run_graph(prompt, graph, config)
        # Display assistant response with icon
        st.markdown(
            f"<div class='chat-bubble assistant'><span class='icon'>🤖</span><div>{response}</div></div>", 
            unsafe_allow_html=True
        )
        st.session_state.messages.append({
            "role": "assistant",
            "content": response
        })
else:
    st.info("Please upload a document to start chatting!")

st.markdown("</div>", unsafe_allow_html=True)
