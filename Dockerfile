FROM python:3.11-slim

WORKDIR /app

# Install dependencies (now much lighter without TF/PyTorch)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy backend code and converted ONNX models
COPY . .

# Koyeb default port is 8000
ENV PORT=8000
EXPOSE 8000

CMD ["sh", "-c", "uvicorn app.main:app --host 0.0.0.0 --port ${PORT}"]
