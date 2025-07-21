
def exact_match(predicted: str, ground_truth: str):
    return int(predicted.strip().lower() == ground_truth.strip().lower())
