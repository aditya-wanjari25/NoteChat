# Welcome to my Custom Agentic RAG App

This is an interactive Agentic Retrieval-Augmented Generation (RAG) app that lets you have conversations with your documents. Agents determine when to query or not query the vector database based on user's input. This project is implemented in LangGraph; it comes with a in-memory checkpointer which enables us to maintain chat history for a single conversation using threads. Pinecone is used as a vector database with OpenAI's `text-embedding-3-large` embeddings. 


Steps to run:
1. Create a virtual environment
2. Install requirements.txt `pip install requirements.txt -r`
3. Create a `.env` file and store all required keys.
4. Run the file: `streamlit run app.py`

How to use:
* On the side bar, upload the documents and click add to vector db.
* Now the Agent is ready to chat!
* Once done, if a document is not needed we can click delete vectors on the side bar.

<img width="1505" alt="image" src="https://github.com/user-attachments/assets/fef723be-77eb-493f-be56-e812b3735e3c" />

<img width="1503" alt="image" src="https://github.com/user-attachments/assets/2b298946-36e2-47f5-91d5-8ec4c90eb710" />


