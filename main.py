from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
import os
import json
import re
import traceback
import fitz  # PyMuPDF
from huggingface_hub import InferenceClient

app = FastAPI(title="MediChain API", version="1.0.3")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

HF_API_KEY = os.environ.get("HF_API_KEY")
if not HF_API_KEY:
    print("CRITICAL ERROR: HF_API_KEY is not set.")

# Use a model that's available on free Hugging Face Inference API
HF_MODEL_NAME = os.environ.get("HF_MODEL_NAME", "meta-llama/Llama-2-7b-chat-hf")

print(f"Using Hugging Face model: {HF_MODEL_NAME}")
print(f"Initializing Inference Client...")

# Initialize the HF client
hf_client = InferenceClient(model=HF_MODEL_NAME, token=HF_API_KEY)

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
        print(f"Calling HF model: {HF_MODEL_NAME}")
        response = hf_client.text_generation(
            prompt=prompt,
            max_new_tokens=1024,
            temperature=0.7,
            do_sample=True,
            top_p=0.95
        )
        
        print(f"HF Response received: {str(response)[:200]}")
        
        if not response:
            raise ValueError("Empty response from Hugging Face")
            
        return response
    except Exception as e:
        print("="*60)
        print(f"HUGGING FACE FAILED | {type(e).__name__}: {e}")
        print(traceback.format_exc())
        print("="*60)
        raise HTTPException(status_code=502, detail={
            "error": "Hugging Face API call failed",
            "exception_type": type(e).__name__,
            "exception_msg": str(e),
            "model": HF_MODEL_NAME,
            "common_causes": [
                "1. Model not accessible → verify HF_API_KEY is correct and has access",
                "2. HF_API_KEY wrong/missing → regenerate at huggingface.co/settings/tokens",
                "3. Gated model → accept license at https://huggingface.co/meta-llama/Llama-2-7b-chat-hf",
                "4. Rate limited → free tier has limits, wait before retrying",
                "5. Model loading on first request → retry in 60 seconds",
                "6. Network/firewall issue → check outbound HTTPS to huggingface.co"
            ]
        })

@app.get("/hf-test", tags=["Meta"])
def hf_test():
    """Call this first to confirm API key works before testing PDF."""
    try:
        text = call_huggingface('Reply with only valid JSON: {"status":"hf_ok"}')
        return {"hf_reachable": True, "raw_response": text[:300], "model": HF_MODEL_NAME}
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
        raise HTTPException(500, {"error": "Invalid JSON from Hugging Face", "parse_error": str(e), "hf_output": raw[:500]})

@app.get("/health", tags=["Meta"])
def health():
    return {"status": "ok", "api_key_configured": bool(HF_API_KEY), "model": HF_MODEL_NAME}

@app.get("/", tags=["Meta"])
def root():
    return {"message": "MediChain API is running", "model": HF_MODEL_NAME, "docs": "/docs"}
