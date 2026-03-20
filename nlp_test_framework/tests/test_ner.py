from models.ner_model import extract_entities
from utils.metrics import compute_f1, entities_to_labels

def test_ner_returns_results(note, golden_entities):
    """Ogni nota deve produrre almeno un'entità."""
    entities = extract_entities(note["text"])
    assert isinstance(entities, list)
    assert len(entities) > 0, f"Nessuna entità trovata in {note['id']}"

def test_ner_f1_above_threshold(note, golden_entities):
    """F1 score vs golden set deve superare soglia minima."""
    predicted = extract_entities(note["text"])
    expected  = golden_entities.get(note["id"], [])

    if not expected:
        pytest.skip(f"Nessun golden set per {note['id']}")

    metrics = compute_f1(
        entities_to_labels(predicted),
        entities_to_labels(expected)
    )
    assert metrics["f1"] >= 0.5, (
        f"F1 troppo basso per {note['id']}: {metrics}"
    )