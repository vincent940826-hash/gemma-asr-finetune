from jiwer import cer

def calculate_cer(references: list[str], predictions: list[str]) -> float:
    """Calculate Character Error Rate (CER)"""
    return cer(references, predictions)
