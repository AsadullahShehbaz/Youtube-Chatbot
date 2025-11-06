# app/main.py

from fastapi import FastAPI, HTTPException, Query
from app.utils.youtube_loader import fetch_transcript
from app.core.vectorstore import build_or_load_vectorstore
from app.core.llm_chain import create_rag_chain
from app.core.config import OPENROUTER_API_KEY, OPENROUTER_BASE_URL
from fastapi.middleware.cors import CORSMiddleware



# Initialize FastAPI app
app = FastAPI(
    title="🎥 RAG YouTube Assistant",
    description="Ask questions about any YouTube video using LangChain + FastAPI.",
    version="1.0.0"
)

@app.get("/")
def home():
    """
    Simple home route to check if API is running.
    """
    return {"message": "✅ Welcome to the RAG YouTube Assistant API!"}

@app.get("/ask/")
def ask_youtube_question(
    video_id: str = Query(..., description="YouTube Video ID (e.g., vJOGC8QJZJQ)"),
    question: str = Query(..., description="Question to ask about the video")
):
    """
    Main endpoint:
    1️⃣ Fetch YouTube transcript
    2️⃣ Build or load FAISS vectorstore
    3️⃣ Create RAG chain
    4️⃣ Generate answer from context
    """
    try:
        # Step 1: Fetch transcript
        transcript = fetch_transcript(video_id)
        if not transcript:
            raise HTTPException(status_code=404, detail="Transcript not found or unavailable.")

        # Step 2: Build or load FAISS
        vectorstore = build_or_load_vectorstore(transcript)

        # Step 3: Build LangChain pipeline
        rag_chain = create_rag_chain(vectorstore, OPENROUTER_API_KEY, OPENROUTER_BASE_URL)

        # Step 4: Run query through the RAG chain
        answer = rag_chain.invoke(question)

        return {
            "video_id": video_id,
            "question": question,
            "answer": answer
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # safer: specify your extension ID later
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
