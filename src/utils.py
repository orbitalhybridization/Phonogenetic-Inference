"""
Utility functions shared across all pipeline scripts.
"""

import os
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple
from collections import Counter


def ensure_directory(path: Path) -> None:
    """Create directory if it doesn't exist."""
    path.mkdir(parents=True, exist_ok=True)


def get_language_input_dir(language: str) -> Path:
    """Get input directory for a specific language."""
    from config import INPUT_DIR
    lang_dir = INPUT_DIR / language
    return lang_dir


def get_language_phones_dir(language: str) -> Path:
    """Get phones output directory for a specific language."""
    from config import PHONES_DIR
    lang_dir = PHONES_DIR / language
    ensure_directory(lang_dir)
    return lang_dir


def get_language_summaries_dir(language: str) -> Path:
    """Get phone_summaries output directory for a specific language."""
    from config import PHONE_SUMMARIES_DIR
    lang_dir = PHONE_SUMMARIES_DIR / language
    ensure_directory(lang_dir)
    return lang_dir


def get_language_vectors_dir(language: str) -> Path:
    """Get vectors output directory for a specific language."""
    from config import VECTORS_DIR
    lang_dir = VECTORS_DIR / language
    ensure_directory(lang_dir)
    return lang_dir


def save_phone_sequence(phones: List[str], language: str, wav_filename: str) -> Path:
    """
    Save recognized phone sequence to .npy file.
    
    Args:
        phones: List of phone symbols (strings)
        language: Language name
        wav_filename: Original WAV filename (e.g., 'sample.wav')
    
    Returns:
        Path to saved .npy file
    """
    output_dir = get_language_phones_dir(language)
    
    # Replace .wav extension with .npy
    npy_filename = wav_filename.replace(".wav", "") + ".npy"
    output_path = output_dir / npy_filename
    
    # Convert to numpy array
    phones_array = np.array(phones, dtype=object)
    np.save(str(output_path), phones_array, allow_pickle=True)
    
    return output_path


def load_phone_sequences(language: str) -> Dict[str, np.ndarray]:
    """
    Load all phone sequences for a language from .npy files.
    
    Args:
        language: Language name
    
    Returns:
        Dictionary mapping filename -> phone array
    """
    lang_dir = get_language_phones_dir(language)
    sequences = {}
    
    if not lang_dir.exists():
        return sequences
    
    for npy_file in lang_dir.glob("*.npy"):
        sequences[npy_file.stem] = np.load(str(npy_file), allow_pickle=True)
    
    return sequences


def aggregate_phone_sequences(language: str) -> List[str]:
    """
    Load and concatenate all phone sequences for a language.
    
    Args:
        language: Language name
    
    Returns:
        Single list of all phones for the language (concatenated)
    """
    sequences = load_phone_sequences(language)
    all_phones = []
    
    for seq in sequences.values():
        all_phones.extend(seq.tolist() if isinstance(seq, np.ndarray) else seq)
    
    return all_phones


def extract_ngrams(phones: List[str], n: int) -> List[str]:
    """
    Extract n-grams from a phone sequence.
    
    Args:
        phones: List of phones
        n: Size of n-gram
    
    Returns:
        List of n-gram strings (e.g., ['p_a', 'a_r', 'r_u'] for bigrams)
    """
    ngrams = []
    for i in range(len(phones) - n + 1):
        ngram = "_".join(phones[i : i + n])
        ngrams.append(ngram)
    return ngrams


def compute_phone_frequencies(phones: List[str]) -> Tuple[Counter, int]:
    """
    Compute frequencies of phones.
    
    Args:
        phones: List of phones
    
    Returns:
        Tuple of (Counter object, total phone count)
    """
    counter = Counter(phones)
    total = sum(counter.values())
    return counter, total


def validate_phone_sequence(phones: List[str], min_length: int = 1) -> bool:
    """
    Validate that a phone sequence is non-empty and reasonable.
    
    Args:
        phones: List of phones
        min_length: Minimum required length
    
    Returns:
        True if valid, False otherwise
    """
    return isinstance(phones, (list, np.ndarray)) and len(phones) >= min_length


def normalize_vector(vector: np.ndarray, method: str = "l2") -> np.ndarray:
    """
    Normalize a vector using specified method.
    
    Args:
        vector: 1D numpy array
        method: 'l2' for L2 norm, None for no normalization
    
    Returns:
        Normalized vector
    """
    if method == "l2":
        norm = np.linalg.norm(vector)
        if norm > 0:
            return vector / norm
    return vector


def pairwise_kl_divergence(matrix: np.ndarray) -> np.ndarray:
    """
    Compute pairwise KL divergence between rows (language vectors).
    
    Args:
        matrix: Shape (n_languages, n_features), each row sums to 1
    
    Returns:
        Distance matrix of shape (n_languages, n_languages)
    """
    n = len(matrix)
    distances = np.zeros((n, n))
    
    for i in range(n):
        for j in range(i + 1, n):
            # KL(P||Q) = sum(P * log(P / Q))
            # Avoid log(0) by adding small epsilon
            eps = 1e-10
            p = matrix[i] + eps
            q = matrix[j] + eps
            
            kl = np.sum(p * np.log(p / q))
            distances[i, j] = kl
            distances[j, i] = kl
    
    return distances


def print_step(step_num: int, message: str) -> None:
    """Print a formatted step message."""
    print(f"\n{'='*70}")
    print(f"STEP {step_num}: {message}")
    print(f"{'='*70}")


def print_info(message: str) -> None:
    """Print an info message with indentation."""
    print(f"  ℹ  {message}")


def print_success(message: str) -> None:
    """Print a success message."""
    print(f"  ✓  {message}")


def print_error(message: str) -> None:
    """Print an error message."""
    print(f"  ✗  {message}")


def print_summary(title: str, items: Dict[str, any]) -> None:
    """Print a summary section."""
    print(f"\n{title}:")
    print("-" * 70)
    for key, value in items.items():
        print(f"  {key:<40} {value}")
