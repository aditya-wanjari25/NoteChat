# Welcome to NoteChat!

NoteChat is an interactive Retrieval-Augmented Generation (RAG) app that lets you have conversations with your documents. Simply upload your file, ask any question, and watch the AI fetch relevant context from your notes in real-time.

This RAG system uses Streamlit for the user interface and file upload, leveraging AWS S3 for storage. It processes documents with PyPDFLoader and splits them into manageable chunks using RecursiveCharacterTextSplitter. The chunks are embedded via OpenAI’s model and indexed in a Pinecone vector database for efficient similarity search. When a query is received, relevant document fragments are retrieved and passed as context to a ChatOpenAI (GPT-4o-mini) model to generate an informed response.






