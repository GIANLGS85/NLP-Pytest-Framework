import allure
import pytest
from nlp_test_framework.models.relation_model import extract_relations, extract_medical_entities, classify_relation

# ── Test data ─────────────────────────────────────────────────────────────────

ENTITY_EXTRACTION_CASES = [
    {
        "id":           "drug_disease",
        "text":         "Ibuprofen 400mg was prescribed for chest pain.",
        "expected_label": "TREATMENT",
        "description":  "Known drug must be labeled as TREATMENT"
    },
    {
        "id":           "diagnosis",
        "text":         "Patient was diagnosed with hypertension.",
        "expected_label": "PROBLEM",
        "description":  "Known disease must be labeled as PROBLEM"
    },
]

UNRELATED_ENTITIES_CASES = [
    {
        "id":       "hospital_monday",
        "text":     "The patient visited the hospital on Monday.",
        "entity_a": "hospital",
        "entity_b": "Monday"
    },
    {
        "id":       "doctor_weather",
        "text":     "The doctor checked in while it was raining.",
        "entity_a": "doctor",
        "entity_b": "raining"
    },
]

# ── Entity extraction tests ───────────────────────────────────────────────────

@allure.epic("NLP Pipeline Testing")
@allure.feature("Relation Extraction")
class TestRelations:

    @allure.story("Medical entity extraction")
    @allure.severity(allure.severity_level.CRITICAL)
    @allure.title("At least one medical entity found in clinical sentence")
    def test_medical_entities_extracted(self):
        text = "Ibuprofen 400mg was prescribed for chest pain."

        with allure.step("Extracting medical entities"):
            entities = extract_medical_entities(text)

        allure.attach(
            f"Text:     {text}\n"
            f"Entities: {entities}",
            name="Extracted Medical Entities",
            attachment_type=allure.attachment_type.TEXT
        )

        with allure.step("Asserting at least one entity was found"):
            assert len(entities) > 0, "No medical entities found"

    @allure.story("Medical entity extraction")
    @allure.severity(allure.severity_level.CRITICAL)
    @allure.title("Expected entity label detected in clinical text")
    @pytest.mark.parametrize(
        "case",
        ENTITY_EXTRACTION_CASES,
        ids=[c["id"] for c in ENTITY_EXTRACTION_CASES]
    )
    def test_entity_label_detected(self, case):
        with allure.step(f"Extracting entities from: '{case['text']}'"):
            entities = extract_medical_entities(case["text"])
            labels   = [e["label"] for e in entities]

        allure.attach(
            f"Text:           {case['text']}\n"
            f"Expected label: {case['expected_label']}\n"
            f"Found labels:   {labels}\n"
            f"All entities:   {entities}",
            name="Entity Label Detection",
            attachment_type=allure.attachment_type.TEXT
        )

        with allure.step(f"Asserting '{case['expected_label']}' is in extracted labels"):
            assert case["expected_label"] in labels, (
                f"{case['description']}. Got: {entities}"
            )

    # ── Relation extraction tests ─────────────────────────────────────────────

    @allure.story("Relation extraction from golden set")
    @allure.severity(allure.severity_level.CRITICAL)
    @allure.title("Each golden case produces at least one relation")
    def test_relations_extracted(self, golden_relations):
        for case in golden_relations:
            with allure.step(f"Extracting relations from: {case['id']}"):
                relations = extract_relations(case["text"])

            allure.attach(
                f"Text:      {case['text']}\n"
                f"Relations: {relations}",
                name=f"Relations — {case['id']}",
                attachment_type=allure.attachment_type.TEXT
            )

            assert len(relations) > 0, (
                f"No relations found for: {case['id']}"
            )

    @allure.story("Relation structure validation")
    @allure.severity(allure.severity_level.NORMAL)
    @allure.title("Every relation dict contains required keys with valid types")
    def test_relation_structure(self, golden_relations):
        required_keys = ["entity_a", "entity_b", "relation", "score"]

        for case in golden_relations:
            with allure.step(f"Validating relation structure for: {case['id']}"):
                relations = extract_relations(case["text"])

            for rel in relations:
                allure.attach(
                    f"Relation: {rel}",
                    name=f"Structure check — {case['id']}",
                    attachment_type=allure.attachment_type.TEXT
                )

                for key in required_keys:
                    assert key in rel, (
                        f"Missing key '{key}' in relation: {rel}"
                    )

                assert isinstance(rel["score"], float), (
                    f"Score must be a float, got: {type(rel['score'])}"
                )
                assert 0.0 <= rel["score"] <= 1.0, (
                    f"Score out of range [0.0, 1.0]: {rel['score']}"
                )

    @allure.story("Drug-Disease relation detection")
    @allure.severity(allure.severity_level.BLOCKER)
    @allure.title("Classic Drug→Disease 'treats' relation is identified")
    def test_treats_relation_detected(self):
        text = "Ibuprofen was prescribed to treat chest pain."

        with allure.step("Extracting relations from clinical sentence"):
            relations      = extract_relations(text)
            relation_types = [r["relation"] for r in relations]

        allure.attach(
            f"Text:             {text}\n"
            f"Relations found:  {relations}\n"
            f"Relation types:   {relation_types}",
            name="Treats Relation Detection",
            attachment_type=allure.attachment_type.TEXT
        )

        with allure.step("Asserting 'treats' relation was extracted"):
            assert "treats" in relation_types, (
                f"Expected 'treats' relation. Got: {relations}"
            )

    @allure.story("Relation recall against golden set")
    @allure.severity(allure.severity_level.CRITICAL)
    @allure.title("Relation type recall meets minimum threshold of 0.5")
    def test_golden_relation_types_match(self, golden_relations):
        total_expected = 0
        total_matched  = 0

        for case in golden_relations:
            with allure.step(f"Evaluating relations for: {case['id']}"):
                relations = extract_relations(case["text"])
                predicted = {r["relation"] for r in relations}
                expected  = {r["relation"] for r in case["expected_relations"]}

                matched = len(predicted & expected)  # True Positives (TP)
                total_expected += len(expected)
                total_matched  += matched

                allure.attach(
                    f"Case:      {case['id']}\n"
                    f"Predicted: {predicted}\n"
                    f"Expected:  {expected}\n"
                    f"Matched:   {matched}/{len(expected)}",
                    name=f"Recall breakdown — {case['id']}",
                    attachment_type=allure.attachment_type.TEXT
                )

        recall = total_matched / total_expected if total_expected > 0 else 0.0

        allure.attach(
            f"Total expected: {total_expected}\n"
            f"Total matched:  {total_matched}\n"
            f"Recall:         {recall:.2f}",
            name="Overall Recall Summary",
            attachment_type=allure.attachment_type.TEXT
        )

        assert recall >= 0.5, (
            f"Relation type recall (the proportion of expected relations "
            f"the model correctly identified) too low: {recall:.2f} "
            f"({total_matched}/{total_expected} matched)"
        )

    @allure.story("Unrelated entity handling")
    @allure.severity(allure.severity_level.NORMAL)
    @allure.title("Unrelated entities in the same sentence produce no meaningful relation")
    @pytest.mark.parametrize(
        "case",
        UNRELATED_ENTITIES_CASES,
        ids=[c["id"] for c in UNRELATED_ENTITIES_CASES]
    )
    def test_no_relation_for_unrelated_entities(self, case):
        with allure.step(
            f"Classifying relation between "
            f"'{case['entity_a']}' and '{case['entity_b']}'"
        ):
            result = classify_relation(case["text"], case["entity_a"], case["entity_b"])

        allure.attach(
            f"Text:       {case['text']}\n"
            f"Entity A:   {case['entity_a']}\n"
            f"Entity B:   {case['entity_b']}\n"
            f"Relation:   {result['relation']}\n"
            f"Score:      {result['score']}",
            name="Unrelated Entity Classification",
            attachment_type=allure.attachment_type.TEXT
        )

        with allure.step("Asserting relation is 'no relation'"):
            assert result["relation"] == "no relation", (
                f"Expected 'no relation' for unrelated entities "
                f"'{case['entity_a']}' and '{case['entity_b']}', "
                f"got: {result}"
            )
