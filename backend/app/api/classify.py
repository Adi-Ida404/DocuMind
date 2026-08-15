from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from backend.app.services.classification import classification_service

router = APIRouter()


class ClassifyRequest(BaseModel):
    text: str


class ClassifyResponse(BaseModel):
    category: str
    confidence: float


@router.post("/classify", response_model=ClassifyResponse)
def classify(request: ClassifyRequest):
    try:
        result = classification_service.classify_text(request.text)
        return result
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))