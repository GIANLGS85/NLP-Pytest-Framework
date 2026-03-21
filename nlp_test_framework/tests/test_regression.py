from nlp_test_framework.models.ner_model import extract_entities
from nlp_test_framework.utils.metrics import entities_to_labels

BASELINE_RESULTS = {
    "note_001": ["john smith|PER", "12/03/2024|DATE"],
    "note_002": ["emily carter|PER", "new york general hospital|ORG"],
}

def test_no_regression_vs_baseline(note):
    predicted = entities_to_labels(extract_entities(note["text"]))
    baseline  = BASELINE_RESULTS.get(note["id"], [])
    for expected_entity in baseline:
        assert expected_entity in predicted, (
            f"Regression detected in {note['id']}: '{expected_entity}' no longer extracted"
        )