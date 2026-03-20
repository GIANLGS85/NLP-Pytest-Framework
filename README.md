# NLP Test Framework

A Python-based test framework for NLP pipelines, built with **pytest** and **HuggingFace Transformers**. Designed as a self-learning project to practice testing real NLP tasks such as Named Entity Recognition, De-identification, and Text Classification.

---

## Features

- **NER testing** — entity extraction with F1 score validation against a golden dataset
- **De-identification** — PHI detection and removal using Microsoft Presidio
- **Text Classification** — zero-shot classification with configurable labels
- **Auto model download** — HuggingFace models are downloaded automatically on first run and cached locally
- **Parametrized tests** — test cases are driven by JSON data files, no hardcoding required
- **Session-scoped fixtures** — models are loaded once per test session for fast execution

---

## Project Structure

The project uses a standard layered architecture to ensure high readability and scalability:

```
nlp_test_framework/
├── models/
│   ├── __init__.py
│   ├── ner_model.py          # HuggingFace NER pipeline wrapper
│   ├── deid_model.py         # Presidio PHI detection and anonymization
│   └── classifier_model.py  # Zero-shot text classification
├── utils/
│   ├── __init__.py
│   ├── metrics.py            # precision, recall, f1 for NLP evaluation
│   └── data_loader.py        # JSON data loading and validation
├── tests/
│   ├── test_ner.py           # Entity extraction tests
│   ├── test_deid.py          # PHI removal tests
│   └── test_classification.py
├── data/
│   ├── patient_notes.json    # Sample input texts
│   └── golden_entities.json  # Expected entities for F1 evaluation
├── conftest.py               # Fixtures, parametrization, setup/teardown
├── pytest.ini                # Pytest configuration
└── requirements.txt
```

---

## Tech Stack

| Component | Library | Purpose |
|---|---|---|
| Test runner | `pytest` | Test discovery, fixtures, parametrization |
| NER | `dslim/bert-base-NER` (HuggingFace) | Named Entity Recognition |
| De-identification | `presidio-analyzer` + `presidio-anonymizer` | PHI detection and removal |
| Classification | `facebook/bart-large-mnli` (HuggingFace) | Zero-shot text classification |
| Metrics | custom `utils/metrics.py` | Precision, Recall, F1 |

---

## Installation

```bash
# Clone the repository
git clone https://github.com/your-username/nlp-test-framework.git
cd nlp-test-framework

# Create and activate a virtual environment
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

> **Note:** HuggingFace models are downloaded automatically on first run. Make sure you have an internet connection and ~2GB of free disk space.

---

## Running Tests

```bash
# Run all tests
pytest

# Run a specific test file
pytest tests/test_ner.py

# Run with verbose output
pytest -v

# Run and stop on first failure
pytest -x
```

Example output:

```
tests/test_ner.py::test_ner_returns_results[note_001] PASSED
tests/test_ner.py::test_ner_returns_results[note_002] PASSED
tests/test_ner.py::test_ner_f1_above_threshold[note_001] PASSED
tests/test_deid.py::test_deidentification_removes_names PASSED
tests/test_classification.py::test_urgent_note_classified_correctly PASSED
```

---

## Data Files

### `data/patient_notes.json`
Contains the input texts used across all test suites. Each note has a unique `id`, a `text` field, and a `category` label.

```json
[
  {
    "id": "note_001",
    "text": "Patient John Smith, 45 years old, visited on 12/03/2024.",
    "category": "routine"
  }
]
```

### `data/golden_entities.json`
Contains the expected entities for each note, used to compute F1 score in NER tests.

```json
{
  "note_001": [
    {"text": "John Smith", "label": "PER"},
    {"text": "12/03/2024", "label": "DATE"}
  ]
}
```

---

## Models

### NER — `dslim/bert-base-NER`
Extracts entities of type `PER`, `ORG`, `LOC`, and `MISC` from free text.

### De-identification — Microsoft Presidio
Detects and anonymizes PHI (Personal Health Information) including names, dates, phone numbers, email addresses, and SSNs.

### Classification — `facebook/bart-large-mnli`
Zero-shot classifier — no fine-tuning needed. Default labels: `urgent`, `routine`, `follow-up`, `administrative`. Labels are fully configurable at runtime.

---

## Configuration

`pytest.ini` controls test discovery and output:

```ini
[pytest]
testpaths = tests
log_cli = true
log_cli_level = INFO
addopts = -v --tb=short
```

---

## Contributing

This is a self-learning project. Contributions, suggestions, and issues are welcome — feel free to open a PR or start a discussion.

---

## License

MIT
