from presidio_analyzer import AnalyzerEngine
from presidio_anonymizer import AnonymizerEngine
from functools import lru_cache

@lru_cache(maxsize=1)
def load_engines():
    analyzer = AnalyzerEngine()
    anonymizer = AnonymizerEngine()
    return analyzer, anonymizer

def deidentify(text: str, language: str = "en") -> str:
    analyzer, anonymizer = load_engines()
    results = analyzer.analyze(text=text, language=language)
    anonymized = anonymizer.anonymize(text=text, analyzer_results=results)
    return anonymized.text

def detect_phi(text: str, language: str = "en") -> list[dict]:
    analyzer, _ = load_engines()
    results = analyzer.analyze(text=text, language=language)
    return [
        {
            "type": r.entity_type,
            "start": r.start,
            "end": r.end,
            "score": round(r.score, 4)
        }
        for r in results
    ]