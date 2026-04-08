#!/usr/bin/env python3
"""
Script 3: Vectorization
========================
Create feature vectors from recognized phones and n-grams.

Input: Recognized phones from output/phones/[language]/
Output: Vector representations in output/vectors/[language]/

Outputs:
  - {language}/{phones,2grams,3grams,4grams,combined}.npy
  - results/{language}/feature_stats.txt

Usage:
    python 3_vectorization.py              # Vectorize all languages
    python 3_vectorization.py --lang garifuna  # Vectorize specific language
"""

import sys
from pathlib import Path
from typing import Dict, Tuple, List

# Add src directory to path
sys.path.insert(0, str(Path(__file__).parent))

import numpy as np
from collections import Counter
from sklearn.preprocessing import normalize

import config
import utils


def extract_ngrams_from_sequence(phones: List[str], n: int) -> List[tuple]:
    """Extract n-grams from a phone sequence."""
    ngrams = []
    for i in range(len(phones) - n + 1):
        ngram = tuple(phones[i : i + n])
        ngrams.append(ngram)
    return ngrams


def create_phone_vector(aggregate_phones: List[str]) -> Tuple[np.ndarray, int]:
    """
    Create normalized phone frequency vector.
    
    Args:
        aggregate_phones: All phones for the language
    
    Returns:
        Tuple of (vector, total_phones)
    """
    if not aggregate_phones:
        return np.array([]), 0
    
    counter = Counter(aggregate_phones)
    total = len(aggregate_phones)
    
    # Get all unique phones, sorted for consistency
    phones = sorted(counter.keys())
    
    # Create frequency vector normalized by total phones
    vector = np.array([counter[p] / total for p in phones], dtype=np.float32)
    
    return vector, total


def create_ngram_vectors(
    aggregate_phones: List[str],
) -> Dict[int, np.ndarray]:
    """
    Create normalized n-gram frequency vectors.
    
    Args:
        aggregate_phones: All phones for the language
    
    Returns:
        Dictionary mapping n -> vector for n in [2, 3, 4]
    """
    ngram_vectors = {}
    total_phones = len(aggregate_phones)
    
    for n in [2, 3, 4]:
        ngrams = extract_ngrams_from_sequence(aggregate_phones, n)
        
        if not ngrams:
            ngram_vectors[n] = np.array([], dtype=np.float32)
            continue
        
        counter = Counter(ngrams)
        ngrams_sorted = sorted(counter.keys())
        
        # Create frequency vector normalized by total phones
        vector = np.array(
            [counter[ng] / total_phones for ng in ngrams_sorted],
            dtype=np.float32
        )
        ngram_vectors[n] = vector
    
    return ngram_vectors


def create_combined_vector(
    phone_vector: np.ndarray,
    ngram_vectors: Dict[int, np.ndarray],
    normalize_features: bool = True,
    apply_weighting: bool = True,
) -> np.ndarray:
    """
    Combine phone and ngram vectors into a single feature vector.
    
    Args:
        phone_vector: Phone frequency vector
        ngram_vectors: Dictionary of n-gram vectors
        normalize_features: Whether to L2-normalize each feature type
        apply_weighting: Whether to apply equal weighting (0.25 each)
    
    Returns:
        Combined feature vector
    """
    components = [phone_vector]
    
    if normalize_features:
        # L2 normalize each component
        phone_vector = utils.normalize_vector(phone_vector, method="l2")
        components = [phone_vector]
        
        for n in [2, 3, 4]:
            ng_vec = utils.normalize_vector(ngram_vectors[n], method="l2")
            if apply_weighting:
                ng_vec = 0.25 * ng_vec
            components.append(ng_vec)
        
        # Also weight the phone vector
        components[0] = 0.25 * components[0]
    else:
        components = [phone_vector]
        for n in [2, 3, 4]:
            components.append(ngram_vectors[n])
    
    # Concatenate all components
    combined = np.concatenate(components)
    
    return combined


def vectorize_language(language: str) -> Tuple[bool, Dict]:
    """
    Create feature vectors for a single language.
    
    Args:
        language: Language name
    
    Returns:
        Tuple of (success, stats_dict)
    """
    # Load all phone sequences for the language
    all_phones = utils.aggregate_phone_sequences(language)
    
    if not all_phones:
        utils.print_error(f"No phone sequences found for {language}")
        return False, {}
    
    # Create phone vector
    phone_vector, total_phones = create_phone_vector(all_phones)
    
    # Create n-gram vectors
    ngram_vectors = create_ngram_vectors(all_phones)
    
    # Create combined vector
    combined_vector = create_combined_vector(
        phone_vector,
        ngram_vectors,
        normalize_features=True,
        apply_weighting=True,
    )
    
    # Save vectors
    output_dir = utils.get_language_vectors_dir(language)
    
    np.save(str(output_dir / "phones.npy"), phone_vector)
    np.save(str(output_dir / "2grams.npy"), ngram_vectors[2])
    np.save(str(output_dir / "3grams.npy"), ngram_vectors[3])
    np.save(str(output_dir / "4grams.npy"), ngram_vectors[4])
    np.save(str(output_dir / "combined.npy"), combined_vector)
    
    # Save statistics
    stats_file = output_dir / "feature_stats.txt"
    with open(stats_file, "w") as f:
        f.write(f"Feature Vectors for {language.capitalize()}\n")
        f.write("=" * 70 + "\n\n")
        
        f.write(f"Total phones aggregated: {total_phones}\n")
        f.write(f"Phone vector shape: {phone_vector.shape}\n")
        f.write(f"2-gram vector shape: {ngram_vectors[2].shape}\n")
        f.write(f"3-gram vector shape: {ngram_vectors[3].shape}\n")
        f.write(f"4-gram vector shape: {ngram_vectors[4].shape}\n")
        f.write(f"Combined vector shape: {combined_vector.shape}\n\n")
        
        f.write("Vector COMPOSITION:\n")
        f.write("-" * 70 + "\n")
        f.write(f"  Phones:   {phone_vector.shape[0]:6d} features (0.25 weighted)\n")
        f.write(f"  2-grams:  {ngram_vectors[2].shape[0]:6d} features (0.25 weighted)\n")
        f.write(f"  3-grams:  {ngram_vectors[3].shape[0]:6d} features (0.25 weighted)\n")
        f.write(f"  4-grams:  {ngram_vectors[4].shape[0]:6d} features (0.25 weighted)\n")
        f.write(f"            {'-' * 6}\n")
        f.write(f"  TOTAL:    {combined_vector.shape[0]:6d} features\n\n")
        
        f.write("Vector NORMALIZATION:\n")
        f.write("-" * 70 + "\n")
        f.write(f"  Each feature type L2-normalized separately\n")
        f.write(f"  Then weighted equally (0.25 each)\n")
        f.write(f"  Combined vector L2 norm: {np.linalg.norm(combined_vector):.6f}\n")
    
    return True, {
        "total_phones": total_phones,
        "n_phones": phone_vector.shape[0],
        "n_2grams": ngram_vectors[2].shape[0],
        "n_3grams": ngram_vectors[3].shape[0],
        "n_4grams": ngram_vectors[4].shape[0],
        "n_combined": combined_vector.shape[0],
    }


def validate_all_language_vectors() -> bool:
    """Validate that all languages have vector files."""
    all_valid = True
    
    for language in config.LANGUAGES:
        vectors_dir = utils.get_language_vectors_dir(language)
        
        required_files = ["phones.npy", "2grams.npy", "3grams.npy", "4grams.npy", "combined.npy"]
        
        for file in required_files:
            if not (vectors_dir / file).exists():
                utils.print_error(f"{language}: Missing {file}")
                all_valid = False
    
    return all_valid


def main():
    """Main entry point."""
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Create feature vectors from recognized phones"
    )
    parser.add_argument(
        "--lang",
        type=str,
        default=None,
        help=f"Vectorize specific language (default: all). Options: {', '.join(config.LANGUAGES)}"
    )
    
    args = parser.parse_args()
    
    # Determine which languages to process
    languages = [args.lang] if args.lang else config.LANGUAGES
    
    # Validate languages
    invalid_langs = [lang for lang in languages if lang not in config.LANGUAGES]
    if invalid_langs:
        utils.print_error(f"Invalid languages: {', '.join(invalid_langs)}")
        utils.print_info(f"Available languages: {', '.join(config.LANGUAGES)}")
        return 1
    
    utils.print_step(3, "VECTORIZATION")
    utils.print_info(f"Languages to process: {', '.join(languages)}")
    
    vectorization_stats = {}
    
    for language in languages:
        success, stats = vectorize_language(language)
        
        if not success:
            utils.print_error(f"{language}: Vectorization failed")
            continue
        
        vectorization_stats[language] = stats
        
        utils.print_success(
            f"{language}: Combined vector shape {stats['n_combined']}"
        )
    
    # Print summary
    if vectorization_stats:
        utils.print_summary("VECTORIZATION SUMMARY", {
            k: f"phones={v['n_phones']}, 2g={v['n_2grams']}, "
               f"3g={v['n_3grams']}, 4g={v['n_4grams']}, "
               f"combined={v['n_combined']}"
            for k, v in vectorization_stats.items()
        })
        
        utils.print_success("Vectorization complete!")
        return 0
    else:
        utils.print_error("No languages vectorized")
        return 1


if __name__ == "__main__":
    exit(main())
