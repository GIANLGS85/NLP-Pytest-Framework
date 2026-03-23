import allure
import pytest
from nlp_test_framework.models.ner_model import extract_entities

NEGATED_NOTES = [
    {"text": "Patient denies chest pain.",        "entity": "chest pain"},
    {"text": "No history of hypertension.",       "entity": "hypertension"},
    {"text": "Patient has no fever or coughing.", "entity": "fever"},
]

@allure.epic("NLP Pipeline Testing")
@allure.feature("Negation Detection")
class TestNegation:

    @allure.story("Negated entity suppression")
    @allure.severity(allure.severity_level.BLOCKER)
    @allure.title("Negated clinical entities must not be extracted")
    @pytest.mark.parametrize("case", NEGATED_NOTES, ids=[c["entity"] for c in NEGATED_NOTES])
    def test_negated_entities_not_extracted(self, case):
        with allure.step(f"Extracting entities from: '{case['text']}'"):
            entities = extract_entities(case["text"])
            found = [e["text"].lower() for e in entities]

        allure.attach(
            f"Input text:       {case['text']}\n"
            f"Negated entity:   {case['entity']}\n"
            f"Extracted:        {found}",
            name="Negation Detection Details",
            attachment_type=allure.attachment_type.TEXT
        )

        with allure.step(f"Asserting '{case['entity']}' was NOT extracted"):
            assert case["entity"] not in found, (
                f"Negated entity '{case['entity']}' should not have been extracted. "
                f"Model returned: {found}"
            )
