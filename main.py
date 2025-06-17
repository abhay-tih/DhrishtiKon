from fastapi import FastAPI, File, UploadFile, Form
from fastapi.responses import JSONResponse
from PIL import Image
import shutil
import os
import io
import fitz  # pymupdf
import json
from pydantic import BaseModel
from surya.layout import LayoutPredictor
from doctr.models import ocr_predictor
from transformers import pipeline
from predict_output import predict_output, predict_multiple_output
from fastapi import Depends, HTTPException, Security, status
from fastapi.security.api_key import APIKeyHeader
from typing import List
API_KEY = "AGRIBOT"  # Replace with your actual secret API key
API_KEY_NAME = "X-API-KEY"
api_key_header = APIKeyHeader(name=API_KEY_NAME, auto_error=False)


async def get_api_key(api_key: str = Security(api_key_header)):
    if api_key == API_KEY:
        return api_key
    else:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or missing API Key",
            headers={"WWW-Authenticate": "API Key"},
        )


app = FastAPI()

# Load models at startup
layout_predictor = LayoutPredictor()
ocr_model = ocr_predictor(det_arch='db_resnet50', reco_arch='crnn_vgg16_bn', pretrained=True)
llm_pipe = pipeline("text-generation", model="meta-llama/Meta-Llama-3.1-8B-Instruct", device="cuda:4")

TEMP_DIR = "temp_output_folder"
TEMP_IMAGE = "sample.png"
TEMP_PDF = "temp_file.pdf"

def save_uploaded_file(uploaded_file: UploadFile, save_path: str):
    with open(save_path, "wb") as f:
        shutil.copyfileobj(uploaded_file.file, f)

def convert_pdf_to_images(pdf_path: str):
    doc = fitz.open(pdf_path)
    if os.path.exists(TEMP_DIR):
        shutil.rmtree(TEMP_DIR)
    os.makedirs(TEMP_DIR)
    for page in doc:
        pix = page.get_pixmap()
        pix.save(f"{TEMP_DIR}/{page.number}.png")
    return len(doc)

@app.post("/predict/")
async def predict(
    file: UploadFile = File(...),
    question: str = Form(...),
    model_type: str = Form("MGVG"),
    api_key: str = Depends(get_api_key)
):
    file_ext = file.filename.split(".")[-1].lower()

    document_type = "image"
    image = None

    if file_ext == "pdf":
        save_uploaded_file(file, TEMP_PDF)
        pages = convert_pdf_to_images(TEMP_PDF)
        document_type = "pdf" if pages > 1 else "image"
        page_path = os.path.join(TEMP_DIR, "0.png")
        image = Image.open(page_path).convert("RGB")
        temp_input_path = TEMP_PDF
    else:
        save_uploaded_file(file, TEMP_IMAGE)
        image = Image.open(TEMP_IMAGE).convert("RGB")
        temp_input_path = TEMP_IMAGE

    # Predict
    answer, block_bboxes, line_bboxes, word_bboxes, point_bboxes, current_page = predict_output(
        temp_input_path, question, llm_pipe, layout_predictor, ocr_model, model_type, document_type,
    return_only_answer=True)

    return JSONResponse({
        "question": question,
        "answer": answer
    })

# Define the new Pydantic model for each question
class QuestionItem(BaseModel):
    question: str
    tag: str  # You can change the type if tag is expected to be something else (e.g., int)

class Questions(BaseModel):
    questions: List[QuestionItem]


@app.post("/predict_multiple/")
async def predict(
    file: UploadFile = File(...),
    questions_str: str = Form(...),
    model_type: str = Form("MGVG"),
    api_key: str = Depends(get_api_key)
):
    try:
        # Parse the questions JSON string into Pydantic model
        questions_data = Questions(questions=json.loads(questions_str))
    except Exception as e:
        return JSONResponse({"error": f"Invalid questions format: {e}"})

    file_ext = file.filename.split(".")[-1].lower()

    document_type = "image"
    image = None

    if file_ext == "pdf":
        save_uploaded_file(file, TEMP_PDF)
        pages = convert_pdf_to_images(TEMP_PDF)
        document_type = "pdf" if pages > 1 else "image"
        page_path = os.path.join(TEMP_DIR, "0.png")
        image = Image.open(page_path).convert("RGB")
        temp_input_path = TEMP_PDF
    else:
        save_uploaded_file(file, TEMP_IMAGE)
        image = Image.open(TEMP_IMAGE).convert("RGB")
        temp_input_path = TEMP_IMAGE

    # Extract just the questions, or you can use both question and tag
    questions = [(q.question, q.tag) for q in questions_data.questions]
    # tags = [q.tag for q in questions_data.questions]  # Optional use

    # Predict
    responses = predict_multiple_output(
        temp_input_path,
        questions,
        llm_pipe,
        layout_predictor,
        ocr_model,
        model_type,
        document_type
    )

    return responses

@app.post("/predict_with_grounding/")
async def predict(
    file: UploadFile = File(...),
    question: str = Form(...),
    model_type: str = Form("MGVG")
):
    file_ext = file.filename.split(".")[-1].lower()

    document_type = "image"
    image = None

    if file_ext == "pdf":
        save_uploaded_file(file, TEMP_PDF)
        pages = convert_pdf_to_images(TEMP_PDF)
        document_type = "pdf" if pages > 1 else "image"
        page_path = os.path.join(TEMP_DIR, "0.png")
        image = Image.open(page_path).convert("RGB")
        temp_input_path = TEMP_PDF
    else:
        save_uploaded_file(file, TEMP_IMAGE)
        image = Image.open(TEMP_IMAGE).convert("RGB")
        temp_input_path = TEMP_IMAGE
    
    # Predict
    answer, block_bboxes, line_bboxes, word_bboxes, point_bboxes, current_page = predict_output(
        temp_input_path, question, llm_pipe, layout_predictor, ocr_model, model_type, document_type
    )

    return JSONResponse({
        "question": question,
        "answer": answer,
        "block_bboxes": block_bboxes,
        "line_bboxes": line_bboxes,
        "word_bboxes": word_bboxes,
        "point_bboxes": point_bboxes,
        "current_page": current_page
    })
