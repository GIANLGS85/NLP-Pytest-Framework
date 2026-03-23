import allure
from nlp_test_framework.models.classifier_model import classify

@allure.epic("NLP Pipeline Testing")
@allure.feature("Text Classification")
class TestClassification:

    @allure.story("Label validation")
    @allure.severity(allure.severity_level.NORMAL)
    @allure.title("Classification returns a valid label")
    def test_classification_returns_label(self, note):
        valid_labels = ["urgent", "routine", "follow-up", "administrative"]

        with allure.step(f"Classifying note: {note['id']}"):
            result = classify(note["text"])

        allure.attach(
            f"Text:   {note['text']}\n"
            f"Result: {result}",
            name="Classification Result",
            attachment_type=allure.attachment_type.TEXT
        )

        assert result["label"] in valid_labels

    @allure.story("Urgent detection")
    @allure.severity(allure.severity_level.BLOCKER)
    @allure.title("Urgent notes correctly classified as urgent")
    def test_urgent_note_classified_correctly(self):
        text = "URGENT: Patient experiencing severe chest pain, call doctor immediately."

        with allure.step("Running classifier on urgent note"):
            result = classify(text)

        with allure.step("Asserting label is 'urgent'"):
            assert result["label"] == "urgent", (
                f"Expected 'urgent', got: {result}"
            )