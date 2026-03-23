import allure
import pytest
from nlp_test_framework.models.ner_model import extract_entities
from nlp_test_framework.utils.metrics import compute_f1, entities_to_labels

@allure.epic("NLP Pipeline Testing")
@allure.feature("Named Entity Recognition")
class TestNER:

    @allure.story("Basic entity extraction")
    @allure.severity(allure.severity_level.CRITICAL)
    @allure.title("NER returns at least one entity per note")
    def test_ner_returns_results(self, note, golden_entities):
        with allure.step(f"Extracting entities from note: {note['id']}"):
            entities = extract_entities(note["text"])

        with allure.step("Asserting at least one entity was found"):
            assert len(entities) > 0, f"No entities found in {note['id']}"

        allure.attach(
            str(entities),
            name="Extracted entities",
            attachment_type=allure.attachment_type.TEXT
        )

    @allure.story("F1 score validation")
    @allure.severity(allure.severity_level.NORMAL)
    @allure.title("NER F1 score meets minimum threshold")
    def test_ner_f1_above_threshold(self, note, golden_entities):
        with allure.step("Extracting predicted entities"):
            predicted = extract_entities(note["text"])

        with allure.step("Loading expected entities from golden set"):
            expected = golden_entities.get(note["id"], [])

        if not expected:
            pytest.skip(f"No golden set for {note['id']}")

        with allure.step("Computing F1 score"):
            metrics = compute_f1(
                entities_to_labels(predicted),
                entities_to_labels(expected)
            )

        allure.attach(
            f"Predicted: {entities_to_labels(predicted)}\n"
            f"Expected:  {entities_to_labels(expected)}\n"
            f"Metrics:   {metrics}",
            name="F1 Score Details",
            attachment_type=allure.attachment_type.TEXT
        )

        assert metrics["f1"] >= 0.5, (
            f"F1 too low for {note['id']}: {metrics}"
        )