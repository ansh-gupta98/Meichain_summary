from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional
from google import genai
import os
import json
import re
import fitz  # PyMuPDF

# ── App Setup ────────────────────────────────────────────────────────────────
app = FastAPI(
    title="MediChain API",
    description="Backend for MediChain: Patient summary generation & Lab report PDF → JSON",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Gemini Setup ──────────────────────────────────────────────────────────────
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
client = genai.Client(api_key=GEMINI_API_KEY)


# ── Request / Response Models ─────────────────────────────────────────────────

class PatientTextInput(BaseModel):
    patient_id: str
    patient_name: Optional[str] = "Unknown"
    age: Optional[int] = None
    gender: Optional[str] = None
    raw_text: str  # Free-form text from K-dot about patient history

class SummaryResponse(BaseModel):
    patient_id: str
    patient_name: str
    summary: str
    key_conditions: list[str]
    current_medications: list[str]
    allergies: list[str]
    doctor_notes: str
    urgency_flag: str  # LOW / MODERATE / HIGH


# ── Helpers ───────────────────────────────────────────────────────────────────

def extract_text_from_pdf(pdf_bytes: bytes) -> str:
    """Extract all text from a PDF using PyMuPDF."""
    doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    full_text = ""
    for page in doc:
        full_text += page.get_text()
    doc.close()
    return full_text.strip()


def clean_json_response(text: str) -> str:
    """Strip markdown fences from Gemini JSON response."""
    text = re.sub(r"```json\s*", "", text)
    text = re.sub(r"```\s*", "", text)
    return text.strip()


# ── Endpoint 1: Patient Summary ───────────────────────────────────────────────

@app.post("/summary", response_model=SummaryResponse, tags=["Summary"])
async def generate_patient_summary(data: PatientTextInput):
    """
    Accepts raw patient medical history text from K-dot and returns
    a structured, doctor-readable summary.
    """

    prompt = f"""
You are a medical AI assistant helping doctors quickly understand a patient's history.

Below is raw medical information received from the patient intake system (K-dot):

Patient Name: {data.patient_name}
Age: {data.age if data.age else "Not provided"}
Gender: {data.gender if data.gender else "Not provided"}
Raw Medical Text:
\"\"\"{data.raw_text}\"\"\"

Your task: Extract and return ONLY a valid JSON object (no markdown, no explanation) with:
{{
  "summary": "<2-4 sentence clear clinical summary for the doctor>",
  "key_conditions": ["<condition1>", "<condition2>", ...],
  "current_medications": ["<med1 dose>", "<med2 dose>", ...],
  "allergies": ["<allergy1>", ...],
  "doctor_notes": "<any critical flags, warnings, or observations the doctor must know>",
  "urgency_flag": "<LOW | MODERATE | HIGH based on severity>"
}}

Rules:
- If information is missing, use empty list [] or "Not mentioned".
- urgency_flag must be exactly one of: LOW, MODERATE, HIGH
- Be concise and clinical. This is read by a doctor, not a patient.
- Do NOT include markdown fences in your response.
"""

    try:
        response = client.models.generate_content(
            model="gemini-2.0-flash",
            contents=prompt
        )
        raw = clean_json_response(response.text)
        parsed = json.loads(raw)

        return SummaryResponse(
            patient_id=data.patient_id,
            patient_name=data.patient_name,
            summary=parsed.get("summary", ""),
            key_conditions=parsed.get("key_conditions", []),
            current_medications=parsed.get("current_medications", []),
            allergies=parsed.get("allergies", []),
            doctor_notes=parsed.get("doctor_notes", ""),
            urgency_flag=parsed.get("urgency_flag", "LOW"),
        )

    except json.JSONDecodeError as e:
        raise HTTPException(status_code=500, detail=f"Gemini returned invalid JSON: {str(e)}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Summary generation failed: {str(e)}")


# ── Endpoint 2: PDF Lab Report → JSON ────────────────────────────────────────

@app.post("/pdf-to-json", tags=["PDF Parser"])
async def parse_lab_report_pdf(file: UploadFile = File(...)):
    """
    Accepts a lab report PDF from K-dot and returns structured JSON
    following a standard medical lab report schema.
    """

    if not file.filename.endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are accepted.")

    pdf_bytes = await file.read()

    # Step 1: Extract raw text from PDF
    try:
        raw_text = extract_text_from_pdf(pdf_bytes)
    except Exception as e:
        raise HTTPException(status_code=422, detail=f"PDF text extraction failed: {str(e)}")

    if not raw_text:
        raise HTTPException(status_code=422, detail="PDF appears to be empty or scanned without OCR.")

    # Step 2: Send to Gemini for structured extraction
    prompt = f"""
You are a medical data extraction AI. Below is the raw text extracted from a patient's lab report PDF.

Raw Lab Report Text:
\"\"\"{raw_text}\"\"\"

Your task: Extract and return ONLY a valid JSON object (no markdown, no explanation) following this schema:

{{
  "report_metadata": {{
    "lab_name": "<name of lab or hospital>",
    "report_date": "<DD-MM-YYYY or null>",
    "report_id": "<report/accession number or null>",
    "referring_doctor": "<doctor name or null>",
    "patient_name": "<patient name or null>",
    "patient_age": "<age or null>",
    "patient_gender": "<Male | Female | Other | null>",
    "sample_type": "<Blood | Urine | Stool | Other | null>"
  }},
  "test_results": [
    {{
      "test_name": "<name of test>",
      "value": "<measured value>",
      "unit": "<unit like mg/dL, g/dL, etc.>",
      "reference_range": "<normal range like 70-110>",
      "status": "<Normal | High | Low | Critical>"
    }}
  ],
  "abnormal_flags": ["<list of test names that are outside normal range>"],
  "overall_impression": "<1-2 sentence clinical impression based on results>",
  "follow_up_recommended": true or false
}}

Rules:
- Extract every test result visible in the report.
- status must be exactly one of: Normal, High, Low, Critical
- If a field is not found, use null.
- Do NOT include markdown or explanation.
"""

    try:
        response = client.models.generate_content(
            model="gemini-2.0-flash",
            contents=prompt
        )
        raw = clean_json_response(response.text)
        parsed = json.loads(raw)
        return parsed

    except json.JSONDecodeError as e:
        raise HTTPException(status_code=500, detail=f"Gemini returned invalid JSON: {str(e)}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Lab report parsing failed: {str(e)}")


# ── Health Check ──────────────────────────────────────────────────────────────

@app.get("/health", tags=["Health"])
def health_check():
    return {"status": "ok", "service": "MediChain API", "version": "1.0.0"}
