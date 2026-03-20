from transformers import pipeline
from functools import lru_cache

@lru_cache(maxsize=1)
def load_ner_pipeline():
    """Automatic download on first run, then cached."""
    return pipeline(
        "ner",
        model="dslim/bert-base-NER",
        aggregation_strategy="simple"
    )

def extract_entities(text: str) -> list[dict]:
    nlp = load_ner_pipeline()
    results = nlp(text)
    return [
        {
            "text": r["word"],
            "label": r["entity_group"],
            "score": round(r["score"], 4)
        }
        for r in results
    ]