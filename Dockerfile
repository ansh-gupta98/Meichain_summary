FROM python:3.11-slim

WORKDIR /app

COPY . .

# Install all required libraries
RUN pip install fastapi uvicorn huggingface_hub PyMuPDF python-multipart

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
