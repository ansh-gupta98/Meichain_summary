from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
import os
import json
import re
import traceback
import fitz  # PyMuPDF
from google import genai

app = FastAPI(title="MediChain API", version="1.0.3")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY")
if not GEMINI_API_KEY:
    print("CRITICAL ERROR: GEMINI_API_KEY is not set.")
client = genai.Client(api_key=GEMINI_API_KEY)

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
        raise ValueError("Empty Gemini response")
    m = re.search(r"```json\s*(\{.*?\})\s*```", text, re.DOTALL)
    if m: return m.group(1).strip()
    m = re.search(r"```\s*(\{.*?\})\s*```", text, re.DOTALL)
    if m: return m.group(1).strip()
    s, e = text.find("{"), text.rfind("}")
    if s != -1 and e > s: return text[s:e+1].strip()
    raise ValueError(f"No JSON found: {text[:200]}")

def call_gemini(prompt: str) -> str:
    try:
        resp = client.models.generate_content(model="gemini-2.0-flash", contents=prompt)
        return resp.text
    except Exception as e:
        print("="*60)
        print(f"GEMINI FAILED | {type(e).__name__}: {e}")
        print(traceback.format_exc())
        print("="*60)
        raise HTTPException(status_code=502, detail={
            "error": "Gemini API call failed",
            "exception_type": type(e).__name__,
            "exception_msg": str(e),
            "common_causes": [
                "1. GEMINI_API_KEY wrong/missing → Railway Variables tab",
                "2. Key invalid/expired → regenerate at aistudio.google.com/apikey",
                "3. Free quota hit (15 req/min) → wait 60s",
                "4. Network blocked → Railway cant reach generativelanguage.googleapis.com"
            ]
        })

@app.get("/gemini-test", tags=["Meta"])
def gemini_test():
    """Call this first to confirm API key works before testing PDF."""
    try:
        text = call_gemini('Reply with only: {"status":"gemini_ok"}')
        return {"gemini_reachable": True, "raw_response": text[:300]}
    except HTTPException as e:
        return {"gemini_reachable": False, "detail": e.detail}

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

    raw = call_gemini(prompt)
    try:
        return json.loads(clean_json_response(raw))
    except (ValueError, json.JSONDecodeError) as e:
        raise HTTPException(500, {"error": "Invalid JSON from Gemini", "parse_error": str(e), "gemini_output": raw[:500]})

@app.get("/health", tags=["Meta"])
def health():
    return {"status": "ok", "api_key_configured": bool(GEMINI_API_KEY)}
