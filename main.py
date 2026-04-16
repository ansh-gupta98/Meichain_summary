from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
import os, json, re, traceback
import fitz  # PyMuPDF
import google.generativeai as genai

# ── App setup ────────────────────────────────────────────────────────────────
app = FastAPI(title="MediChain API", version="2.0.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], allow_methods=["*"], allow_headers=["*"]
)

# ── Gemini setup (lazy — won't crash on startup if key missing) ───────────────
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY", "AIzaSyDBISr99uV9tRxaunA_AHo7G_Za9LsPJls")
_model = None  # initialized on first request, not at import time

def get_model():
    global _model
    if _model is None:
        if not GEMINI_API_KEY:
            raise HTTPException(
                status_code=500,
                detail="GEMINI_API_KEY environment variable not set on Railway."
            )
        genai.configure(api_key=GEMINI_API_KEY)
        _model = genai.GenerativeModel("gemini-3-flash-preview")
        print("Gemini model initialized.")
    return _model

# ── Helpers ───────────────────────────────────────────────────────────────────
def extract_text_from_pdf(pdf_bytes: bytes) -> str:
    try:
        doc = fitz.open(stream=pdf_bytes, filetype="pdf")
        text = "".join(page.get_text() for page in doc)
        doc.close()
        return text.strip()
    except Exception as e:
        raise Exception(f"PyMuPDF error: {str(e)}")


def clean_json_response(text: str) -> str:
    """Strip markdown fences and extract the first {...} block."""
    if not text:
        raise ValueError("Empty Gemini response")
    # Try ```json ... ``` block first
    m = re.search(r"```json\s*(\{.*?\})\s*```", text, re.DOTALL)
    if m:
        return m.group(1).strip()
    # Try plain ``` ... ``` block
    m = re.search(r"```\s*(\{.*?\})\s*```", text, re.DOTALL)
    if m:
        return m.group(1).strip()
    # Fallback: first { ... } in the string
    s, e = text.find("{"), text.rfind("}")
    if s != -1 and e > s:
        return text[s:e + 1].strip()
    raise ValueError(f"No JSON object found in response: {text[:300]}")


def call_gemini(prompt: str) -> str:
    """Call Gemini and return raw text. Raises HTTPException on failure."""
    try:
        model = get_model()
        response = model.generate_content(prompt)
        raw = response.text
        if not raw:
            raise ValueError("Gemini returned empty content.")
        return raw
    except HTTPException:
        raise  # already formatted
    except Exception as e:
        print("=" * 60)
        print(f"GEMINI FAILED | {type(e).__name__}: {e}")
        print(traceback.format_exc())
        print("=" * 60)
        raise HTTPException(status_code=502, detail={
            "error": "Gemini API call failed",
            "exception_type": type(e).__name__,
            "exception_msg": str(e),
            "common_causes": [
                "GEMINI_API_KEY not set or invalid",
                "Exceeded free-tier quota — check console.cloud.google.com",
                "Network issue between Railway and Google APIs",
            ]
        })


# ── Routes ────────────────────────────────────────────────────────────────────

@app.get("/", tags=["Meta"])
def root():
    return {
        "message": "MediChain API is running",
        "model": "gemini-2.0-flash",
        "docs": "/docs"
    }


@app.get("/health", tags=["Meta"])
def health():
    return {
        "status": "ok",
        "api_key_configured": bool(GEMINI_API_KEY),
        "model": "gemini-3-flash-preview"
    }


@app.get("/gemini-test", tags=["Meta"])
def gemini_test():
    """Quick smoke-test — call this before /pdf-to-json to confirm Gemini works."""
    try:
        raw = call_gemini('Reply with ONLY valid JSON, no markdown: {"status":"gemini_ok"}')
        return {
            "gemini_reachable": True,
            "raw_response": raw[:300],
            "model": "gemini-2.0-flash"
        }
    except HTTPException as e:
        return {"gemini_reachable": False, "detail": e.detail}


@app.post("/pdf-to-json", tags=["PDF Parser"])
async def parse_lab_report_pdf(file: UploadFile = File(...)):
    if not (file.filename or "").lower().endswith(".pdf"):
        raise HTTPException(400, "Only PDF files accepted.")

    pdf_bytes = await file.read()
    if not pdf_bytes:
        raise HTTPException(400, "Empty file uploaded.")

    # Extract text
    try:
        raw_text = extract_text_from_pdf(pdf_bytes)
    except Exception as e:
        raise HTTPException(422, f"Text extraction failed: {e}")

    if not raw_text or len(raw_text) < 10:
        raise HTTPException(
            422,
            "PDF appears to be a scanned image with no extractable text. "
            "Please upload a text-based PDF."
        )

    prompt = f"""You are a medical data extraction assistant.
Extract all lab report data from the text below and return ONLY a valid JSON object.
Do NOT include markdown formatting, code fences, or any explanation — pure JSON only.

Lab Report Text:
{raw_text[:6000]}

Required JSON structure (use null for any missing fields):
{{
  "report_metadata": {{
    "lab_name": null,
    "patient_name": null,
    "patient_age": null,
    "patient_gender": null,
    "date": null,
    "report_id": null
  }},
  "test_results": [
    {{
      "test_name": "string",
      "value": "string",
      "unit": null,
      "reference_range": null,
      "status": "Normal or High or Low or Unknown"
    }}
  ],
  "abnormal_flags": [],
  "overall_impression": "string"
}}"""

    raw = call_gemini(prompt)

    try:
        cleaned = clean_json_response(raw)
        return json.loads(cleaned)
    except (ValueError, json.JSONDecodeError) as e:
        raise HTTPException(500, {
            "error": "Gemini returned invalid JSON",
            "parse_error": str(e),
            "gemini_output_preview": raw[:500]
        })
