# -------------------------------------------------------------
# 🐳 Dockerfile for FastAPI + LangChain (Beginner Friendly)
# -------------------------------------------------------------

# 1️⃣ Use an official Python image as base
FROM python:3.13.4-slim

# 2️⃣ Set working directory in the container
WORKDIR /app

# 3️⃣ Copy requirements first (for caching layers)
COPY requirements.txt .

# 4️⃣ Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# 5️⃣ Copy the entire project into the container
COPY . .

# 6️⃣ Expose port (FastAPI default: 8000)
EXPOSE 8000

# 7️⃣ Run the FastAPI app using uvicorn
# --host 0.0.0.0 allows access from outside container
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
