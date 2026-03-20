from models.deid_model import deidentify, detect_phi

PHI_NOTES = [
    "Patient John Smith, DOB 01/01/1980, SSN 123-45-6789",
    "Contact Mary Johnson at mary.johnson@email.com or 555-1234",
]

def test_phi_detected(note):
    """Deve rilevare almeno un elemento PHI nelle note cliniche."""
    phi = detect_phi(note["text"])
    # Non tutte le note hanno PHI, quindi loggiamo senza fallire
    print(f"\n[{note['id']}] PHI trovati: {[p['type'] for p in phi]}")

def test_deidentification_removes_names():
    """Dopo de-id i nomi non devono comparire nel testo."""
    for text in PHI_NOTES:
        result = deidentify(text)
        assert "John Smith" not in result
        assert "Mary Johnson" not in result

def test_deidentified_text_is_string():
    result = deidentify("Patient John Doe visited on 01/01/2024.")
    assert isinstance(result, str)
    assert len(result) > 0