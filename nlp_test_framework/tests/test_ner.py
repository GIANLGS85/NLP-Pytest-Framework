import pytest
from models.ner_model import extract_entities
from utils.metrics import compute_f1, entities_to_labels

def test_ner_returns_results(note, golden_entities):
    """Each note must produce at least one identity."""
    entities = extract_entities(note["text"])
    assert isinstance(entities, list)
    assert len(entities) > 0, f"no entity found in {note['id']}"

def test_ner_f1_above_threshold(note, golden_entities):
    """F1 score vs golden set must exceed the minimum threshold."""
    predicted = extract_entities(note["text"])
    print(f"\nPredicted: {predicted}")
    expected  = golden_entities.get(note["id"], [])
    print(f"\nexpected: {expected}")

    if not expected:
        pytest.skip(f"No golden set found for {note['id']}")

    metrics = compute_f1(
        entities_to_labels(predicted),
        entities_to_labels(expected)
    )
    assert metrics["f1"] >= 0.5, (
        f"F1 too low for {note['id']}: {metrics}"
    )