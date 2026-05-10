# backend/Dockerfile
FROM python:3.10-slim
WORKDIR /app

RUN apt-get update && apt-get install -y \
    ffmpeg git libsndfile1 && rm -rf /var/lib/apt/lists/*

# ab ye kaam karega, kyunki context root hai (.)
COPY requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt

# backend ka source copy karo
COPY backend/ .

EXPOSE 8001
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8001"]
