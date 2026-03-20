def compute_f1(predicted: list[str], expected: list[str]) -> dict:
    pred_set = set(predicted)
    exp_set = set(expected)

    tp = len(pred_set & exp_set)
    fp = len(pred_set - exp_set)
    fn = len(exp_set - pred_set)

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall    = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1        = (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0

    return {"precision": precision, "recall": recall, "f1": f1}

def entities_to_labels(entities: list[dict]) -> list[str]:
    """Normalizes the entity dict list in a list of strings 'text|label'."""
    return [f"{e['text'].lower()}|{e['label']}" for e in entities]