from transformers import pipeline
from functools import lru_cache

LABELS = ["urgent", "routine", "follow-up", "administrative"]

@lru_cache(maxsize=1)
def load_classifier():
    return pipeline(
        "zero-shot-classification",
        model="facebook/bart-large-mnli"
    )

def classify(text: str, labels: list[str] = None) -> dict:
    clf = load_classifier()
    candidate_labels = labels or LABELS
    result = clf(text, candidate_labels)
    return {
        "label": result["labels"][0],
        "score": round(result["scores"][0], 4),
        "all": dict(zip(result["labels"], result["scores"]))
    }