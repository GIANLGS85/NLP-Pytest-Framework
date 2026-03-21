from nlp_test_framework.models.ner_model import extract_entities

NEGATED_NOTES = [
    {"text": "Patient denies chest pain.",        "entity": "chest pain"},
    {"text": "No history of hypertension.",       "entity": "hypertension"},
    {"text": "Patient has no fever or coughing.", "entity": "fever"},
]

def test_negated_entities_not_extracted():
    for case in NEGATED_NOTES:
        entities = extract_entities(case["text"])
        found = [e["text"].lower() for e in entities]
        assert case["entity"] not in found, (
            f"Negated entity '{case['entity']}' should not be extracted"
        )
