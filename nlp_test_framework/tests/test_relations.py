import pytest
from models.relation_model import extract_relations, extract_medical_entities

# ── Basic entity extraction tests ─────────────────────────────────────────────

def test_medical_entities_extracted():
    """At least one medical entity must be found in a clinical sentence."""
    text     = "Ibuprofen 400mg was prescribed for chest pain."
    entities = extract_medical_entities(text)
    assert len(entities) > 0, "No medical entities found"

def test_treatment_entity_detected():
    """A known drug name must be labeled as TREATMENT."""
    text     = "Ibuprofen 400mg was prescribed for chest pain."
    entities = extract_medical_entities(text)
    labels   = [e["label"] for e in entities]
    assert "TREATMENT" in labels, f"No TREATMENT entity found. Got: {entities}"

def test_problem_entity_detected():
    """A known disease/symptom must be labeled as PROBLEM."""
    text     = "Patient was diagnosed with hypertension."
    entities = extract_medical_entities(text)
    labels   = [e["label"] for e in entities]
    assert "PROBLEM" in labels, f"No PROBLEM entity found. Got: {entities}"

# ── Relation extraction tests ──────────────────────────────────────────────────

def test_relations_extracted(golden_relations):
    """Each golden case must produce at least one relation."""
    for case in golden_relations:
        relations = extract_relations(case["text"])
        assert len(relations) > 0, (
            f"No relations found for: {case['id']}"
        )

def test_relation_structure(golden_relations):
    """Every relation dict must contain required keys with valid types."""
    for case in golden_relations:
        relations = extract_relations(case["text"])
        for rel in relations:
            assert "entity_a" in rel
            assert "entity_b" in rel
            assert "relation" in rel
            assert "score"    in rel
            assert isinstance(rel["score"], float)
            assert 0.0 <= rel["score"] <= 1.0

def test_treats_relation_detected():
    """
    Classic Drug→Disease 'treats' relation must be identified.
    This is the most important relation type for clinical NLP (Natural Language Processing).
    """
    text      = "Ibuprofen was prescribed to treat chest pain."
    relations = extract_relations(text)
    relation_types = [r["relation"] for r in relations]
    assert "treats" in relation_types, (
        f"Expected 'treats' relation. Got: {relations}"
    )

def test_golden_relation_types_match(golden_relations):
    """
    Predicted relation types must match the golden set.
    Uses F1 (F1 Score) logic: counts True Positives (TP) across all cases.
    """
    total_expected = 0
    total_matched  = 0

    for case in golden_relations:
        relations  = extract_relations(case["text"])
        predicted  = {r["relation"] for r in relations}
        expected   = {r["relation"] for r in case["expected_relations"]}

        total_expected += len(expected)
        total_matched  += len(predicted & expected)  # True Positives (TP)

    recall = total_matched / total_expected if total_expected > 0 else 0.0
    assert recall >= 0.5, (
        f"Relation type recall too low: {recall:.2f} "
        f"({total_matched}/{total_expected} matched)"
    )

def test_no_relation_for_unrelated_entities():
    """
    Two unrelated entities in the same sentence should not produce
    a meaningful relation — score for 'no relation' should be highest.
    """
    from models.relation_model import classify_relation
    text   = "The patient visited the hospital on Monday."
    result = classify_relation(text, "hospital", "Monday")
    assert result["relation"] == "no relation", (
        f"Expected 'no relation', got: {result}"
    )