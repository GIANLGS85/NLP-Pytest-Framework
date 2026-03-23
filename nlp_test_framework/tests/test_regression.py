import allure
import pytest
from nlp_test_framework.models.ner_model import extract_entities
from nlp_test_framework.utils.metrics import entities_to_labels

# Working baseline results for the dslim/bert-base-NER model.
# Reflects what the model actually produces based on its training data and capabilities.
# Note: dslim/bert-base-NER only supports PER, ORG, LOC, MISC labels.
# DATE, TIME, MONEY are NOT supported — use Jean-Baptiste/roberta-large-ner-english
# if those entity types are needed.
BASELINE_RESULTS = {
    "note_001": ["john smith|PER"],
    "note_002": ["emily carter|PER", "new york general hospital|ORG"],
}

@allure.epic("NLP Pipeline Testing")
@allure.feature("Regression Testing")
class TestRegression:

    @allure.story("Baseline entity extraction")
    @allure.severity(allure.severity_level.BLOCKER)
    @allure.title("Model output must match the established baseline")
    def test_no_regression_vs_baseline(self, note):
        with allure.step(f"Extracting entities from note: {note['id']}"):
            predicted = entities_to_labels(extract_entities(note["text"]))

        with allure.step("Loading baseline for this note"):
            baseline = BASELINE_RESULTS.get(note["id"], [])

        if not baseline:
            pytest.skip(
                f"No baseline defined for {note['id']} — "
                f"add it to BASELINE_RESULTS once model output is confirmed"
            )

        allure.attach(
            f"Note ID:   {note['id']}\n"
            f"Text:      {note['text']}\n\n"
            f"Predicted: {predicted}\n"
            f"Baseline:  {baseline}\n\n"
            f"Missing:   {[e for e in baseline if e not in predicted]}",
            name="Regression Comparison",
            attachment_type=allure.attachment_type.TEXT
        )

        with allure.step("Asserting all baseline entities are still extracted"):
            missing = []
            for expected_entity in baseline:
                if expected_entity not in predicted:
                    missing.append(expected_entity)

            assert not missing, (
                f"Regression detected in {note['id']}:\n"
                f"  Predicted: {predicted}\n"
                f"  Missing:   {missing}\n\n"
                f"This means the model no longer extracts entities "
                f"it previously could. Check if the model or its "
                f"configuration has changed."
            )

    @allure.story("Baseline coverage")
    @allure.severity(allure.severity_level.NORMAL)
    @allure.title("All notes in the test suite have a defined baseline")
    def test_all_notes_have_baseline(self, note):
        """
        Ensures the baseline (the reference output used to detect
        regressions) is kept in sync with the data files as new
        notes are added.
        """
        with allure.step(f"Checking baseline coverage for: {note['id']}"):
            has_baseline = note["id"] in BASELINE_RESULTS

        allure.attach(
            f"Note ID:      {note['id']}\n"
            f"Has baseline: {has_baseline}\n"
            f"Defined IDs:  {list(BASELINE_RESULTS.keys())}",
            name="Baseline Coverage",
            attachment_type=allure.attachment_type.TEXT
        )

        if not has_baseline:
            pytest.fail(
                f"Note '{note['id']}' has no baseline entry in BASELINE_RESULTS. "
                f"Run the model manually, verify the output, then add it to the baseline."
            )
