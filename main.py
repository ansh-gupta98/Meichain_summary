from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional
import os
import json
import re
import fitz  # PyMuPDF
from google import genai

# -- App Setup --
app = FastAPI(
    title="MediChain API",
    description="Backend for MediChain: Patient summary generation & Lab report PDF → JSON",
    version="1.0.1"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# -- Gemini Setup --
# Fetching from env; ensure this is set in Railway Dashboard
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY")

# Stop the app early if key is missing (helps debugging logs)
if not GEMINI_API_KEY:
    print("CRITICAL ERROR: GEMINI_API_KEY is not set in environment variables.")

client = genai.Client(api_key=GEMINI_API_KEY)

# -- Helpers --

def extract_text_from_pdf(pdf_bytes: bytes) -> str:
    """Extract all text from a PDF using PyMuPDF."""
    try:
        doc = fitz.open(stream=pdf_bytes, filetype="pdf")
        full_text = ""
        for page in doc:
            full_text += page.get_text()
        doc.close()
        return full_text.strip()
    except Exception as e:
        raise Exception(f"PyMuPDF failed to process file: {str(e)}")

def clean_json_response(text: str) -> str:
    """
    Robust JSON extraction. 
    Finds the first '{' and the last '}' to strip away any 
    conversational text or markdown fences Gemini might add.
    """
    try:
        match = re.search(r"(\{.*\})", text, re.DOTALL)
        if match:
            return match.group(1)
        return text.strip()
    except Exception:
        return text.strip()

# -- Endpoint: PDF Lab Report → JSON --

@app.post("/pdf-to-json", tags=["PDF Parser"])
async def parse_lab_report_pdf(file: UploadFile = File(...)):
    """
    Accepts a lab report PDF and returns structured JSON.
    """
    if not file.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are accepted.")

    # 1. Read Bytes
    try:
        pdf_bytes = await file.read()
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to read upload: {str(e)}")

    # 2. Extract Text
    try:
        raw_text = extract_text_from_pdf(pdf_bytes)
    except Exception as e:
        raise HTTPException(status_code=422, detail=f"Text extraction failed: {str(e)}")

    if not raw_text or len(raw_text) < 10:
        raise HTTPException(status_code=422, detail="PDF is empty or requires OCR (scanned image).")

    # 3. Gemini Prompt
    prompt = f"""
    Return ONLY a valid JSON object for the following lab report text. 
    Do not include markdown markers or preamble.

    Text: {raw_text[:4000]} 

    Schema:
    {{
      "report_metadata": {{ "lab_name": "string", "patient_name": "string", "date": "string" }},
      "test_results": [ {{ "test_name": "string", "value": "string", "unit": "string", "status": "Normal|High|Low" }} ],
      "abnormal_flags": [],
      "overall_impression": "string"
    }}
    """

    # 4. API Call with Retries (Simple)
    try:
        response = client.models.generate_content(
            model="gemini-2.0-flash",
            contents=prompt
        )
        
        cleaned_json = clean_json_response(response.text)
        parsed_data = json.loads(cleaned_json)
        return parsed_data

    except json.JSONDecodeError:
        # If JSON is garbled, return the raw text for debugging
        raise HTTPException(status_code=500, detail=f"Gemini returned invalid JSON structure: {response.text[:100]}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Gemini API Error: {str(e)}")

@app.get("/health")
def health():
    return {"status": "ok", "api_key_configured": bool(GEMINI_API_KEY)}
