"""
Fuzzy matching utilities for element ID matching

Helps match CSV element IDs to OCR results that may have OCR errors
"""

import logging
from typing import List, Tuple, Optional
from difflib import SequenceMatcher

logger = logging.getLogger(__name__)


def normalize_element_id(element_id: str) -> str:
    """
    Normalize an element ID for comparison.

    Common OCR mistakes:
    - O (letter) ↔ 0 (digit)
    - I (letter) ↔ 1 (digit)
    - l (lowercase L) ↔ 1 (digit)
    - S ↔ 5
    - Z ↔ 2

    Args:
        element_id: Element ID to normalize

    Returns:
        Normalized element ID
    """
    normalized = element_id.upper()

    # Replace common OCR mistakes
    replacements = {
        'O': '0',
        'I': '1',
        'L': '1',
        'S': '5',
        'Z': '2'
    }

    for old, new in replacements.items():
        normalized = normalized.replace(old, new)

    return normalized


def similarity_score(str1: str, str2: str) -> float:
    """
    Calculate similarity score between two strings.

    Args:
        str1: First string
        str2: Second string

    Returns:
        Similarity score between 0.0 and 1.0
    """
    # Normalize both strings
    norm1 = normalize_element_id(str1)
    norm2 = normalize_element_id(str2)

    # Calculate similarity using SequenceMatcher
    return SequenceMatcher(None, norm1, norm2).ratio()


def fuzzy_match_element_id(target_id: str,
                           candidates: List[str],
                           threshold: float = 0.8) -> Optional[Tuple[str, float]]:
    """
    Find best fuzzy match for target element ID in list of candidates.

    Args:
        target_id: Element ID to find (from CSV)
        candidates: List of candidate strings (from OCR)
        threshold: Minimum similarity score to accept (0.0-1.0)

    Returns:
        Tuple of (best_match, score) or None if no good match found
    """
    if not candidates:
        return None

    best_match = None
    best_score = 0.0

    for candidate in candidates:
        score = similarity_score(target_id, candidate)

        if score > best_score:
            best_score = score
            best_match = candidate

    if best_score >= threshold:
        return (best_match, best_score)

    return None


def fuzzy_match_batch(target_ids: List[str],
                     candidates: List[str],
                     threshold: float = 0.8) -> dict:
    """
    Match multiple target IDs to candidates.

    Args:
        target_ids: List of element IDs to find (from CSV)
        candidates: List of candidate strings (from OCR)
        threshold: Minimum similarity score to accept

    Returns:
        Dictionary mapping target_id → (matched_candidate, score)
    """
    matches = {}

    for target_id in target_ids:
        match_result = fuzzy_match_element_id(target_id, candidates, threshold)
        if match_result:
            matches[target_id] = match_result

    return matches


def is_likely_element_id(text: str, min_length: int = 4, max_length: int = 10) -> bool:
    """
    Check if text looks like it could be an element ID.

    Element IDs are typically:
    - 4-10 characters long
    - Mostly digits
    - May have 1-2 letters at the end (e.g., "4081b")

    Args:
        text: Text to check
        min_length: Minimum length for element ID
        max_length: Maximum length for element ID

    Returns:
        True if text looks like an element ID
    """
    if not text:
        return False

    # Check length
    if not (min_length <= len(text) <= max_length):
        return False

    # Must have at least some digits
    if not any(c.isdigit() for c in text):
        return False

    # Count digits vs letters
    num_digits = sum(c.isdigit() for c in text)
    num_letters = sum(c.isalpha() for c in text)

    # Element IDs are mostly digits
    # Allow up to 3 letters (e.g., "4081b" or "87747a")
    if num_letters > 3:
        return False

    # Must be mostly alphanumeric
    num_alnum = num_digits + num_letters
    if num_alnum < len(text) * 0.8:  # At least 80% alphanumeric
        return False

    return True


def filter_likely_element_ids(texts: List[str]) -> List[str]:
    """
    Filter list of texts to only those that look like element IDs.

    Args:
        texts: List of OCR text results

    Returns:
        Filtered list containing only likely element IDs
    """
    return [text for text in texts if is_likely_element_id(text)]
