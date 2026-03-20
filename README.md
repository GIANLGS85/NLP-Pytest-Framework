## Project Structure
The project uses a standard layered architecture to ensure high readability and scalability:
```text
nlp_test_framework/
├── tests/
│   ├── test_ner.py       # Entity extraction
│   ├── test_deid.py      # PHI removal
│   ├── test_classification.py
├── data/
│   ├── patient_notes.json
│   └── golden_entities.json
├── conftest.py
└── pytest.ini
```
