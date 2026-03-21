from transformers import pipeline
from functools import lru_cache
from models.ner_model import load_ner_pipeline

# Relation types we want to detect
RELATION_LABELS = [
    "treats",  # Drug treats Disease
    "causes",  # Drug causes side effect
    "diagnoses",  # Test diagnoses Disease
    "contraindicates",  # Drug contraindicates condition
    "no relation"
]


@lru_cache(maxsize=1)
def load_medical_ner():
    """
    HUMADEX english_medical_ner recognizes:
    - PROBLEM  (diseases, symptoms, conditions)
    - TREATMENT (drugs, therapies)
    - TEST     (diagnostic procedures)
    """
    return pipeline(
        "ner",
        model="HUMADEX/english_medical_ner",
        aggregation_strategy="simple"
    )


@lru_cache(maxsize=1)
def load_relation_classifier():
    """
    Zero-shot classification (ZSC) to label
    the relation between two extracted entities.
    """
    return pipeline(
        "zero-shot-classification",
        model="facebook/bart-large-mnli"
    )


def extract_medical_entities(text: str) -> list[dict]:
    """Extract PROBLEM, TREATMENT, TEST entities from clinical text."""
    nlp = load_medical_ner()
    results = nlp(text)
    return [
        {
            "text": r["word"],
            "label": r["entity_group"],  # PROBLEM | TREATMENT | TEST
            "score": round(r["score"], 4)
        }
        for r in results
    ]


def classify_relation(text: str, entity_a: str, entity_b: str) -> dict:
    """
    Given a sentence and two entities, classify the relation between them.

    Example:
        text     = "Ibuprofen was prescribed for chest pain."
        entity_a = "Ibuprofen"   (TREATMENT)
        entity_b = "chest pain"  (PROBLEM)
        → {"relation": "treats", "score": 0.87}
    """
    clf = load_relation_classifier()

    # Build a hypothesis sentence that gives the model context
    hypothesis_text = (
        f"In this sentence, {entity_a} and {entity_b} are related. "
        f"The full sentence is: {text}"
    )

    result = clf(hypothesis_text, RELATION_LABELS)
    return {
        "entity_a": entity_a,
        "entity_b": entity_b,
        "relation": result["labels"][0],
        "score": round(result["scores"][0], 4)
    }


def extract_relations(text: str) -> list[dict]:
    """
    Full pipeline: extract entities then classify
    relations between each (TREATMENT, PROBLEM) pair.
    """
    entities = extract_medical_entities(text)
    relations = []

    treatments = [e for e in entities if e["label"] == "TREATMENT"]
    problems = [e for e in entities if e["label"] == "PROBLEM"]
    tests = [e for e in entities if e["label"] == "TEST"]

    # Drug → Disease pairs
    for drug in treatments:
        for problem in problems:
            rel = classify_relation(text, drug["text"], problem["text"])
            if rel["relation"] != "no relation":
                relations.append(rel)

    # Test → Disease pairs
    for test in tests:
        for problem in problems:
            rel = classify_relation(text, test["text"], problem["text"])
            if rel["relation"] != "no relation":
                relations.append(rel)

    return relations