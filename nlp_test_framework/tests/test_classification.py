import pytest
from models.classifier_model import classify

def test_classification_returns_label(note):
    """Ogni nota deve ricevere una label."""
    result = classify(note["text"])
    assert "label" in result
    assert result["label"] in ["urgent", "routine", "follow-up", "administrative"]

def test_urgent_note_classified_correctly():
    """Nota con URGENT deve essere classificata come urgent."""
    text = "URGENT: Patient experiencing severe chest pain, call doctor immediately."
    result = classify(text)
    assert result["label"] == "urgent", f"Atteso 'urgent', ottenuto: {result}"

def test_classification_score_range(note):
    """Score di confidenza deve essere tra 0 e 1."""
    result = classify(note["text"])
    assert 0.0 <= result["score"] <= 1.0