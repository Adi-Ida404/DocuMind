"""
Loads the trained model once and exposes a simple predict() function.
This is the ONLY file that should know anything about sklearn internals.
"""
import pickle
from pathlib import Path
from functools import lru_cache

MODEL_PATH = Path(__file__).resolve().parents[1] / "models" / "document_classifier.pkl"


class DocumentClassifier:
    def __init__(self, model_path: Path = MODEL_PATH):
        with open(model_path, "rb") as f:
            self.pipeline = pickle.load(f)

    def predict(self, text: str) -> dict:
        category = self.pipeline.predict([text])[0]
        # predict_proba gives confidence for each class
        proba = self.pipeline.predict_proba([text])[0]
        confidence = float(max(proba))
        return {"category": category, "confidence": round(confidence, 4)}


@lru_cache(maxsize=1)
def get_classifier() -> DocumentClassifier:
    """Load the model once, reuse across requests (avoids reloading pkl every call)."""
    return DocumentClassifier()