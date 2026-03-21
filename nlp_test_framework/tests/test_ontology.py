from nlp_test_framework.utils.ontology import normalize_term


def test_synonym_normalized_correctly():
    assert normalize_term("high blood pressure") == "hypertension"
    assert normalize_term("heart attack") == "myocardial infarction"