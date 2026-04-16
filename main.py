from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
import os
import json
import re
import fitz  # PyMuPDF
from google import genai

# -- App Setup --
app = FastAPI(
    title="MediChain API",
    description="Backend for MediChain: Lab report PDF → JSON",
    version="1.0.2"
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# -- Gemini Setup --
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY")
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
    Multi-strategy JSON extraction — handles all Gemini output formats:
    1. ```json ... ``` fenced blocks
    2. ``` ... ``` plain fenced blocks
    3. Raw { ... } anywhere in the text
    """
    if not text:
        raise ValueError("Empty response from Gemini")

    # Strategy 1: extract from ```json ... ``` fence
    match = re.search(r"```json\s*(\{.*?\})\s*```", text, re.DOTALL)
    if match:
        return match.group(1).strip()

    # Strategy 2: extract from ``` ... ``` plain fence
    match = re.search(r"```\s*(\{.*?\})\s*```", text, re.DOTALL)
    if match:
        return match.group(1).strip()

    # Strategy 3: find first { to last }
    start = text.find("{")
    end   = text.rfind("}")
    if start != -1 and end != -1 and end > start:
        return text[start:end + 1].strip()

    raise ValueError(f"No JSON object found in Gemini response: {text[:300]}")


# -- Endpoint: PDF Lab Report → JSON --

@app.post("/pdf-to-json", tags=["PDF Parser"])
async def parse_lab_report_pdf(file: UploadFile = File(...)):
    """
    Accepts a lab report PDF and returns structured JSON.
    """
    # 1. Validate file type
    filename = file.filename or ""
    if not filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are accepted.")

    # 2. Read bytes
    try:
        pdf_bytes = await file.read()
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to read upload: {str(e)}")

    if not pdf_bytes:
        raise HTTPException(status_code=400, detail="Uploaded file is empty.")

    # 3. Extract text
    try:
        raw_text = extract_text_from_pdf(pdf_bytes)
    except Exception as e:
        raise HTTPException(status_code=422, detail=f"Text extraction failed: {str(e)}")

    if not raw_text or len(raw_text) < 10:
        raise HTTPException(
            status_code=422,
            detail="PDF appears empty or is a scanned image (requires OCR). "
                   "Please upload a text-based PDF."
        )

    # 4. Truncate safely — Gemini 2.0 flash has large context but keep prompt lean
    truncated_text = raw_text[:6000]

    # 5. Build prompt — explicit instruction to return ONLY JSON
    prompt = f"""You are a medical data extraction assistant.
Extract the lab report data from the text below and return ONLY a valid JSON object.
Do NOT include any explanation, markdown, or code fences. Output raw JSON only.

Lab Report Text:
{truncated_text}

Return this exact JSON structure (fill all fields from the report, use null if not found):
{{
  "report_metadata": {{
    "lab_name": "string or null",
    "patient_name": "string or null",
    "patient_age": "string or null",
    "patient_gender": "string or null",
    "date": "string or null",
    "report_id": "string or null"
  }},
  "test_results": [
    {{
      "test_name": "string",
      "value": "string",
      "unit": "string or null",
      "reference_range": "string or null",
      "status": "Normal or High or Low or Unknown"
    }}
  ],
  "abnormal_flags": ["list of test names that are High or Low"],
  "overall_impression": "brief summary string"
}}"""

    # 6. Call Gemini
    raw_response_text = ""
    try:
        response = client.models.generate_content(
            model="gemini-2.0-flash",   # correct model ID for google-genai SDK
            contents=prompt
        )
        raw_response_text = response.text
    except Exception as e:
        raise HTTPException(
            status_code=502,
            detail=f"Gemini API call failed: {str(e)}"
        )

    # 7. Parse JSON from response
    try:
        cleaned = clean_json_response(raw_response_text)
        parsed  = json.loads(cleaned)
        return parsed
    except (ValueError, json.JSONDecodeError) as e:
        # Return debug info so you can see exactly what Gemini sent back
        raise HTTPException(
            status_code=500,
            detail={
                "error":         "Failed to parse Gemini response as JSON",
                "parse_error":   str(e),
                "gemini_output": raw_response_text[:500]   # first 500 chars for debug
            }
        )


# -- Health Check --

@app.get("/health", tags=["Meta"])
def health():
    return {
        "status": "ok",
        "api_key_configured": bool(GEMINI_API_KEY)
    }
