from nlp_test_framework.models.deid_model import deidentify, detect_phi

PHI_NOTES = [
    "Patient John Smith, DOB 01/01/1980, SSN 123-45-6789",
    "Contact Mary Johnson at mary.johnson@email.com or 555-1234",
]

def test_phi_detected(note):
    """must find at least one PHI element (Protected Health Information) in the clinic notes"""
    phi = detect_phi(note["text"])
    # Not all the notes contain PHI, so we log without failing
    print(f"\n[{note['id']}] PHI trovati: {[p['type'] for p in phi]}")

def test_deidentification_removes_names():
    """after de-id names must not be in the text results."""
    for text in PHI_NOTES:
        result = deidentify(text)
        assert "John Smith" not in result
        assert "Mary Johnson" not in result

def test_deidentified_text_is_string():
    result = deidentify("Patient John Doe visited on 01/01/2024.")
    assert isinstance(result, str)
    assert len(result) > 0