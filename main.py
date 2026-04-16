from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
import os
import json
import re
import traceback
import fitz  # PyMuPDF
import requests

app = FastAPI(title="MediChain API", version="1.0.3")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

HF_API_KEY = os.environ.get("HF_API_KEY")
if not HF_API_KEY:
    print("CRITICAL ERROR: HF_API_KEY is not set.")

# Hugging Face API endpoint
HF_API_URL = "google/gemma-4-31B-it"

def extract_text_from_pdf(pdf_bytes: bytes) -> str:
    try:
        doc = fitz.open(stream=pdf_bytes, filetype="pdf")
        text = "".join(page.get_text() for page in doc)
        doc.close()
        return text.strip()
    except Exception as e:
        raise Exception(f"PyMuPDF error: {str(e)}")

def clean_json_response(text: str) -> str:
    if not text:
        raise ValueError("Empty HF response")
    m = re.search(r"```json\s*(\{.*?\})\s*```", text, re.DOTALL)
    if m: return m.group(1).strip()
    m = re.search(r"```\s*(\{.*?\})\s*```", text, re.DOTALL)
    if m: return m.group(1).strip()
    s, e = text.find("{"), text.rfind("}")
    if s != -1 and e > s: return text[s:e+1].strip()
    raise ValueError(f"No JSON found: {text[:200]}")

def call_huggingface(prompt: str) -> str:
    try:
        headers = {"Authorization": f"Bearer {HF_API_KEY}"}
        payload = {
            "inputs": prompt,
            "parameters": {
                "max_new_tokens": 1024,
                "temperature": 0.7
            }
        }
        response = requests.post(HF_API_URL, headers=headers, json=payload, timeout=30)
        response.raise_for_status()
        
        result = response.json()
        # Extract generated text from response
        if isinstance(result, list) and len(result) > 0:
            return result[0].get("generated_text", "")
        return result.get("generated_text", "")
    except Exception as e:
        print("="*60)
        print(f"HUGGING FACE FAILED | {type(e).__name__}: {e}")
        print(traceback.format_exc())
        print("="*60)
        raise HTTPException(status_code=502, detail={
            "error": "Hugging Face API call failed",
            "exception_type": type(e).__name__,
            "exception_msg": str(e),
            "common_causes": [
                "1. HF_API_KEY wrong/missing → set in environment variables",
                "2. Key invalid/expired → regenerate at huggingface.co/settings/tokens",
                "3. Rate limited → wait before retrying",
                "4. Network blocked → check firewall/proxy",
                "5. Model loading → inference API may take time to load model on first request"
            ]
        })

@app.get("/hf-test", tags=["Meta"])
def hf_test():
    """Call this first to confirm API key works before testing PDF."""
    try:
        text = call_huggingface('Reply with only: {"status":"hf_ok"}')
        return {"hf_reachable": True, "raw_response": text[:300]}
    except HTTPException as e:
        return {"hf_reachable": False, "detail": e.detail}

@app.post("/pdf-to-json", tags=["PDF Parser"])
async def parse_lab_report_pdf(file: UploadFile = File(...)):
    if not (file.filename or "").lower().endswith(".pdf"):
        raise HTTPException(400, "Only PDF files accepted.")
    pdf_bytes = await file.read()
    if not pdf_bytes:
        raise HTTPException(400, "Empty file.")
    try:
        raw_text = extract_text_from_pdf(pdf_bytes)
    except Exception as e:
        raise HTTPException(422, f"Text extraction failed: {e}")
    if not raw_text or len(raw_text) < 10:
        raise HTTPException(422, "PDF empty or scanned image — no extractable text.")

    prompt = f"""You are a medical data extraction assistant.
Extract lab report data and return ONLY valid JSON. No markdown, no explanation.

Lab Report Text:
{raw_text[:6000]}

JSON structure (null for missing fields):
{{
  "report_metadata": {{
    "lab_name": null, "patient_name": null, "patient_age": null,
    "patient_gender": null, "date": null, "report_id": null
  }},
  "test_results": [
    {{"test_name": "string", "value": "string", "unit": null,
      "reference_range": null, "status": "Normal or High or Low or Unknown"}}
  ],
  "abnormal_flags": [],
  "overall_impression": "string"
}}"""

    raw = call_huggingface(prompt)
    try:
        return json.loads(clean_json_response(raw))
    except (ValueError, json.JSONDecodeError) as e:
        raise HTTPException(500, {"error": "Invalid JSON from Hugging Face", "parse_error": str(e), "gemini_output": raw[:500]})

@app.get("/health", tags=["Meta"])
def health():
    return {"status": "ok", "api_key_configured": bool(HF_API_KEY)}
