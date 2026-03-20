import pytest
from sklearn.metrics import f1_score


class TestClinicalNER:
    def test_fever_detection(self, clinical_pipeline, patient_notes):
        result = clinical_pipeline.annotate(patient_notes[0]["text"])
        entities = result["entities"]

        fever_entities = [e for e in entities if "fever" in e.lower()]
        assert len(fever_entities) > 0, "No fever entity detected"

    @pytest.mark.parametrize("text, expected_entities", [
        ("Patient reports chest pain", ["SYMPTOM"]),
        ("55 y.o. female diabetes", ["AGE", "GENDER", "PROBLEM"]),
    ])
    def test_ner_accuracy(self, clinical_pipeline, text, expected_entities):
        result = clinical_pipeline.annotate(text)
        found_labels = [ent[1] for ent in result["entities"]]
        score = f1_score(expected_entities, found_labels, average="partial", zero_division=0)
        assert score >= 0.8, f"F1-score: {score}"
