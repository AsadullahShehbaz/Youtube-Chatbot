# app/core/vectorstore.py

from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import RecursiveCharacterTextSplitter
import os

def build_or_load_vectorstore(text: str, persist_dir: str = "data/faiss_index"):
    """
    Build a FAISS vector store from transcript text.
    If vectorstore already exists, it loads it for faster startup.
    """
    # Define the embedding model
    embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    
    # Split long transcript into small overlapping chunks
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks = splitter.create_documents([text])

    # If FAISS already saved locally, load instead of recomputing
    if os.path.exists(persist_dir):
        print("🔁 Loading existing FAISS vectorstore...")
        return FAISS.load_local(persist_dir, embedding_model, allow_dangerous_deserialization=True)

    print("⚙️ Building new FAISS vectorstore...")
    vectorstore = FAISS.from_documents(chunks, embedding_model)
    vectorstore.save_local(persist_dir)
    return vectorstore
