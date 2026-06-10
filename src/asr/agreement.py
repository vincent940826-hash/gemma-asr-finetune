def get_longest_common_prefix(prev_text: str, curr_text: str) -> str:
    """
    Finds the longest common prefix between two strings at the character level.
    Designed for Chinese ASR partial results.
    """
    # [Future] Heuristic: trim unstable tail characters before matching
    # e.g., if we want to drop the last 1-2 characters to avoid flickering
    # trimmed_curr = curr_text[:-1] if len(curr_text) > 2 else curr_text
    # For MVP, we do exact LCP:
    
    i = 0
    min_len = min(len(prev_text), len(curr_text))
    while i < min_len and prev_text[i] == curr_text[i]:
        i += 1
    
    return prev_text[:i]
