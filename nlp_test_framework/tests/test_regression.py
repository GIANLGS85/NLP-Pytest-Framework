from nlp_test_framework.models.ner_model import extract_entities
from nlp_test_framework.utils.metrics import entities_to_labels

# Working Baseline results for the dslim/bert-base-NER model.
# This is what the model should produce for these notes, based on its training data and capabilities.
# Or if you need DATE extraction, switch to a model that supports it `Jean-Baptiste/roberta-large-ner-english`
# which includes DATE, TIME, MONEY entities.

BASELINE_RESULTS = {
    "note_001": ["john smith|PER"],
    "note_002": ["emily carter|PER", "new york general hospital|ORG"],
}

# baseline with failures with unexpected data type for the model dslim/bert-base-NER
# BASELINE_RESULTS = {
#     "note_001": ["john smith|PER", "12/03/2024|DATE"],
#     "note_002": ["emily carter|PER", "new york general hospital|ORG"],
# }
#
# The baseline says note_001 should produce both john smith|PER and 12/03/2024|DATE, but the model only returns john smith|PER. The date is not being extracted at all.
# Root cause
# dslim/bert-base-NER does not have a DATE label. It only recognizes four entity types:
# LabelMeaningPERPersonORGOrganizationLOCLocationMISCMiscellaneous
# Dates are not in its vocabulary. The baseline was written incorrectly — it assumed DATE support that this model simply doesn't have.
#

def test_no_regression_vs_baseline(note):
    predicted = entities_to_labels(extract_entities(note["text"]))
    baseline  = BASELINE_RESULTS.get(note["id"], [])
    for expected_entity in baseline:
        assert expected_entity in predicted, (
            f"Regression detected in {note['id']}: predicted was '{predicted}', expected: '{expected_entity}' no longer extracted"
        )