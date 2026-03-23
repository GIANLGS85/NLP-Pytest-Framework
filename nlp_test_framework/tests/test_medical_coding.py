import allure
import pytest
from nlp_test_framework.models.ner_model import extract_entities

# ICD-10-CM (International Classification of Diseases, 10th Revision, Clinical Modification) codes
CODING_MAP = {
    "hypertension": "I10",
    "chest pain":   "R07.9",
    "depression":   "F32.9",
    "hyperthyroid": "E05.90"
}

@allure.epic("NLP Pipeline Testing")
@allure.feature("Medical Coding")
class TestMedicalCoding:

    @allure.story("ICD-10-CM code mapping")
    @allure.severity(allure.severity_level.CRITICAL)
    @allure.title("Extracted diseases map to a valid ICD-10-CM code")
    def test_disease_maps_to_icd10_code(self, note):
        with allure.step(f"Extracting entities from note: {note['id']}"):
            entities = extract_entities(note["text"])

        with allure.step("Filtering entities labeled as Disease"):
            diseases = [e["text"].lower() for e in entities if e["label"] == "Disease"]

        if not diseases:
            pytest.skip(f"No Disease entities found in {note['id']}")

        allure.attach(
            f"Diseases found: {diseases}\n"
            f"Known ICD-10-CM codes: {CODING_MAP}",
            name="Disease Extraction Details",
            attachment_type=allure.attachment_type.TEXT
        )

        with allure.step("Asserting each disease has a mapped ICD-10-CM code"):
            for disease in diseases:
                allure.attach(
                    f"Disease: '{disease}' → "
                    f"ICD-10-CM: {CODING_MAP.get(disease, 'NOT FOUND')}",
                    name=f"Coding result: {disease}",
                    attachment_type=allure.attachment_type.TEXT
                )
                assert disease in CODING_MAP, (
                    f"No ICD-10-CM (International Classification of Diseases, "
                    f"10th Revision, Clinical Modification) code mapped for: '{disease}'"
                )