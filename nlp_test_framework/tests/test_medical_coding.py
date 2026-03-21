# Maps extracted diseases to ICD-10-CM codes
from nlp_test_framework.models.ner_model import extract_entities

CODING_MAP = {
    "hypertension":   "I10",
    "chest pain":     "R07.9",
    "depression":     "F32.9",
    "hyperthyroid":   "E05.90"
}

def test_disease_maps_to_icd10_code(note):
    entities = extract_entities(note["text"])
    diseases = [e["text"].lower() for e in entities if e["label"] == "Disease"]
    for disease in diseases:
        assert disease in CODING_MAP, f"No ICD-10-CM code mapped for: {disease}"