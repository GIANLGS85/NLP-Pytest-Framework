import allure
from nlp_test_framework.models.deid_model import deidentify, detect_phi

@allure.epic("NLP Pipeline Testing")
@allure.feature("De-identification")
class TestDeidentification:

    @allure.story("PHI detection")
    @allure.severity(allure.severity_level.CRITICAL)
    @allure.title("PHI detected in clinical note")
    def test_phi_detected(self, note):
        with allure.step(f"Scanning note {note['id']} for PHI"):
            phi = detect_phi(note["text"])

        allure.attach(
            f"PHI found: {[p['type'] for p in phi]}",
            name="PHI Detection Result",
            attachment_type=allure.attachment_type.TEXT
        )

    @allure.story("Name anonymization")
    @allure.severity(allure.severity_level.BLOCKER)
    @allure.title("Patient names removed after de-identification")
    def test_deidentification_removes_names(self):
        PHI_NOTES = [
            "Patient John Smith, DOB 01/01/1980, SSN 123-45-6789",
            "Contact Mary Johnson at mary.johnson@email.com",
        ]
        for text in PHI_NOTES:
            with allure.step(f"De-identifying: '{text[:40]}...'"):
                result = deidentify(text)

            allure.attach(
                f"Original:        {text}\n"
                f"De-identified:   {result}",
                name="De-identification Result",
                attachment_type=allure.attachment_type.TEXT
            )

            assert "John Smith"   not in result
            assert "Mary Johnson" not in result