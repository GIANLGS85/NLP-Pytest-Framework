import allure
import pytest
from nlp_test_framework.utils.ontology import normalize_term

# Test cases as data — keeps the test class clean and easy to extend
SYNONYM_CASES = [
    {"input": "high blood pressure", "expected": "hypertension"},
    {"input": "heart attack",        "expected": "myocardial infarction"},
    {"input": "bp",                  "expected": "hypertension"},
    {"input": "MI",                  "expected": "myocardial infarction"},
]

@allure.epic("NLP Pipeline Testing")
@allure.feature("Ontology Normalization")
class TestOntology:

    @allure.story("Synonym normalization")
    @allure.severity(allure.severity_level.CRITICAL)
    @allure.title("Clinical synonyms normalize to standard medical terms")
    @pytest.mark.parametrize(
        "case",
        SYNONYM_CASES,
        ids=[c["input"] for c in SYNONYM_CASES]
    )
    def test_synonym_normalized_correctly(self, case):
        with allure.step(f"Normalizing term: '{case['input']}'"):
            result = normalize_term(case["input"])

        allure.attach(
            f"Input:    {case['input']}\n"
            f"Expected: {case['expected']}\n"
            f"Got:      {result}",
            name="Normalization Result",
            attachment_type=allure.attachment_type.TEXT
        )

        with allure.step(f"Asserting '{case['input']}' → '{case['expected']}'"):
            assert result == case["expected"], (
                f"Synonym not normalized correctly: "
                f"'{case['input']}' → '{result}' "
                f"(expected '{case['expected']}')"
            )

    @allure.story("Unknown term passthrough")
    @allure.severity(allure.severity_level.NORMAL)
    @allure.title("Unknown terms are returned unchanged")
    @pytest.mark.parametrize("term", [
        "appendicitis",
        "tachycardia",
        "unknown_condition"
    ])
    def test_unknown_term_returned_as_is(self, term):
        """
        Terms not in the ontology (the structured vocabulary of
        medical concepts) must pass through unchanged rather than
        being silently dropped or mapped to a wrong term.
        """
        with allure.step(f"Normalizing unknown term: '{term}'"):
            result = normalize_term(term)

        allure.attach(
            f"Input: {term}\n"
            f"Got:   {result}",
            name="Unknown Term Result",
            attachment_type=allure.attachment_type.TEXT
        )

        with allure.step(f"Asserting '{term}' is returned unchanged"):
            assert result == term.lower(), (
                f"Unknown term '{term}' should pass through unchanged, got: '{result}'"
            )
