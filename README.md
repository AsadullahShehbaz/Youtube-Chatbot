Perfect 💪 — you’re doing exactly the right thing.
You already understand LangChain and RAG logic deeply, so now I’ll give you a **complete beginner-friendly FastAPI + LangChain production project**.
It’s clean, fully working, and carefully commented for easy learning.

---

## 🎯 Project Goal

A **RAG YouTube Assistant API** — you send a YouTube video ID and a question → it fetches the transcript, creates vector embeddings, retrieves relevant chunks, and uses LLM to answer.

---

## 📂 Folder Structure

```
rag_youtube/
│
├── app/
│   ├── main.py                # 🚀 FastAPI entry point (start server here)
│   ├── core/
│   │   ├── config.py          # 🌿 Environment & API keys
│   │   ├── vectorstore.py     # 💾 Embedding + FAISS logic
│   │   ├── llm_chain.py       # 🧠 LangChain RAG pipeline
│   └── utils/
│       ├── youtube_loader.py  # 📹 Fetch YouTube transcript
│
├── data/                      # Persistent vectorstore storage
│   └── faiss_index/
│
├── .env                       # 🔑 API keys & settings
├── requirements.txt
└── README.md
```

---

## 1️⃣ `.env`

Put your OpenRouter credentials here (free OpenAI proxy API):

```env
OPENROUTER_API_KEY=your_openrouter_api_key_here
OPENROUTER_BASE_URL=https://openrouter.ai/api/v1
```

---

## 2️⃣ `requirements.txt`

```txt
fastapi
uvicorn
python-dotenv
langchain
langchain-core
langchain-openai
langchain-huggingface
langchain-community
langchain-text-splitters
faiss-cpu
youtube-transcript-api
```

Then install them:

```bash
pip install -r requirements.txt
```

---

## 3️⃣ `app/core/config.py`

Handles environment variables and API key loading.

```python
# app/core/config.py

from dotenv import load_dotenv
import os

# Load variables from .env file
load_dotenv()

# Get API keys from environment
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
OPENROUTER_BASE_URL = os.getenv("OPENROUTER_BASE_URL")

# Optional safety check
if not OPENROUTER_API_KEY or not OPENROUTER_BASE_URL:
    raise ValueError("⚠️ Missing OpenRouter API credentials in .env file")
```

---

## 4️⃣ `app/utils/youtube_loader.py`

Fetches transcript from YouTube video using the `youtube-transcript-api`.

```python
# app/utils/youtube_loader.py

from youtube_transcript_api import YouTubeTranscriptApi, TranscriptsDisabled

def fetch_transcript(video_id: str) -> str:
    """
    Fetches the transcript text from a YouTube video.
    Args:
        video_id (str): YouTube video ID
    Returns:
        str: Combined transcript text
    """
    try:
        transcript_obj = YouTubeTranscriptApi().fetch(video_id, languages=['en'])
        # Combine all text snippets into one string
        transcript = " ".join(snippet.text for snippet in transcript_obj.snippets)
        return transcript
    except TranscriptsDisabled:
        raise Exception("No captions available for this video.")
    except Exception as e:
        raise Exception(f"Error fetching transcript: {e}")
```

---

## 5️⃣ `app/core/vectorstore.py`

Handles text splitting, embedding generation, and FAISS vector store.

```python
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
```

---

## 6️⃣ `app/core/llm_chain.py`

This file defines the **RAG pipeline** (Retriever → Augment → Generate).

```python
# app/core/llm_chain.py

from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableParallel, RunnableLambda, RunnablePassthrough

def create_rag_chain(vectorstore, api_key, api_base):
    """
    Builds the full RAG (Retrieval-Augmented Generation) pipeline
    using LangChain's Runnable interface.
    """

    # Step 1: Retriever - fetch relevant chunks from vectorstore
    retriever = vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": 3})

    # Step 2: LLM - Use OpenRouter (free OpenAI-compatible API)
    llm = ChatOpenAI(
        model="openai/gpt-4o-mini",
        openai_api_key=api_key,
        openai_api_base=api_base
    )

    # Step 3: Prompt template
    prompt = PromptTemplate(
        template = """
        You are a helpful AI assistant.
        Use ONLY the context provided below to answer the question.
        If the answer is not in the context, just say: "I don't know."

        Context:
        {context}

        Question:
        {question}
        """,
        input_variables=['context','question']
    )

    # Helper: format documents into plain text
    def format_docs(docs):
        return "\n\n".join(doc.page_content for doc in docs)

    # Step 4: Create parallel retriever + passthrough pipeline
    parallel_chain = RunnableParallel({
        'context': retriever | RunnableLambda(format_docs),
        'question': RunnablePassthrough()
    })

    # Step 5: Combine everything into one chain
    rag_chain = parallel_chain | prompt | llm | StrOutputParser()
    return rag_chain
```

---

## 7️⃣ `app/main.py`

The main FastAPI entry point.
Handles routes, input, and response formatting.

```python
# app/main.py

from fastapi import FastAPI, HTTPException, Query
from app.utils.youtube_loader import fetch_transcript
from app.core.vectorstore import build_or_load_vectorstore
from app.core.llm_chain import create_rag_chain
from app.core.config import OPENROUTER_API_KEY, OPENROUTER_BASE_URL

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
```

---

## 8️⃣ Run the API 🚀

In the project root:

```bash
uvicorn app.main:app --reload
```

Then open your browser:

```
http://127.0.0.1:8000/docs
```

Here you can **try out the `/ask` endpoint** visually:

```
video_id = vJOGC8QJZJQ
question = What is LangGraph?
```

---

## ✅ Expected Output (Example)

```json
{
  "video_id": "vJOGC8QJZJQ",
  "question": "What is LangGraph?",
  "answer": "LangGraph is an open-source framework that allows building agentic AI systems on top of LangChain."
}
```

---

## 💡 Tips for You as a Beginner

* Always use `/docs` (FastAPI Swagger UI) for testing routes visually.
* `.env` should **never** be shared publicly.
* You can later add:

  * `/upload/` route for local text files (PDF, DOCX, etc.)
  * `/history/` route to view old queries
  * Frontend using **Streamlit** or **React**

---

Would you like me to now add **Dockerfile + Render deployment guide** (so your app runs online for free)?
That’s the next perfect step for making it production-ready.
