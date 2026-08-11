"""
Trains a tiny document classifier: TF-IDF + Logistic Regression.
Run this once to produce ml/models/document_classifier.pkl
"""
import pickle
from pathlib import Path

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

# --- Tiny toy dataset (replace with real data later) ---
TEXTS = [
    "This paper presents a novel neural network architecture for image classification.",
    "We propose a new transformer-based method for natural language understanding.",
    "Our experiments show state-of-the-art results on the benchmark dataset.",
    "Please find attached the invoice for services rendered in March.",
    "Your payment of $500 is due by the end of this month.",
    "This receipt confirms your purchase of office supplies.",
    "Once upon a time, in a small village, there lived a young girl.",
    "The dragon soared over the mountains as the knights prepared for battle.",
    "She opened the old book and began reading the story to her children.",
]
LABELS = [
    "research", "research", "research",
    "finance", "finance", "finance",
    "story", "story", "story",
]

def train():
    pipeline = Pipeline([
        ("tfidf", TfidfVectorizer(stop_words="english")),
        ("clf", LogisticRegression(max_iter=1000)),
    ])
    pipeline.fit(TEXTS, LABELS)

    out_path = Path(__file__).resolve().parents[1] / "models" / "document_classifier.pkl"
    out_path.parent.mkdir(parents=True, exist_ok=True)

    with open(out_path, "wb") as f:
        pickle.dump(pipeline, f)

    print(f"Model saved to {out_path}")

if __name__ == "__main__":
    train()