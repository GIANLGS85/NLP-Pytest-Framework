import pytest
from utils.data_loader import load_json
from models.ner_model import load_ner_pipeline
from models.deid_model import load_engines
from models.classifier_model import load_classifier

# ── Setup/teardown globale ─────────────────────────────────────────────────────

def pytest_configure(config):
    print("\n[setup] Loading NLP models (first run may download)...")

def pytest_sessionfinish(session, exitstatus):
    print("\n[teardown] Test session complete.")

# ── Fixtures: dati ────────────────────────────────────────────────────────────

@pytest.fixture(scope="session")
def patient_notes():
    return load_json("patient_notes.json")

@pytest.fixture(scope="session")
def golden_entities():
    return load_json("golden_entities.json")

# ── Fixtures: modelli (session scope = caricati una volta sola) ───────────────

@pytest.fixture(scope="session")
def ner_pipeline():
    return load_ner_pipeline()

@pytest.fixture(scope="session")
def deid_engines():
    return load_engines()

@pytest.fixture(scope="session")
def classifier():
    return load_classifier()

# ── Parametrizzazione dinamica ────────────────────────────────────────────────

def pytest_generate_tests(metafunc):
    """Parametrizza automaticamente i test che usano 'note'."""
    if "note" in metafunc.fixturenames:
        notes = load_json("patient_notes.json")
        metafunc.parametrize("note", notes, ids=[n["id"] for n in notes])