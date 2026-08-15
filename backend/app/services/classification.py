from ml.inference.predictor import get_classifier


class ClassificationService:
    def __init__(self):
        self.classifier = get_classifier()

    def classify_text(self, text: str) -> dict:
        if not text or not text.strip():
            raise ValueError("Text cannot be empty")
        return self.classifier.predict(text)


# Simple singleton — backend imports this, never touches ml/ directly
classification_service = ClassificationService()