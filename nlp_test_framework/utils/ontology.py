SYNONYMS = {
    "high blood pressure": "hypertension",
    "bp":                  "hypertension",
    "heart attack":        "myocardial infarction",
    "MI":                  "myocardial infarction",
}

def normalize_term(term: str) -> str:
    return SYNONYMS.get(term.lower(), term.lower())
