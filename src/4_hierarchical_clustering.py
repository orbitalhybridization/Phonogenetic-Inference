#!/usr/bin/env python3
"""
Script 4: Hierarchical Clustering
==================================
Build hierarchical clustering dendrogram from language vectors.

Input: Combined vectors from output/vectors/[language]/combined.npy
Output: Dendrogram in output/results/trees/

Usage:
    python 4_hierarchical_clustering.py              # Build dendrogram from all languages
    python 4_hierarchical_clustering.py --linkage ward  # Specify linkage method
    python 4_hierarchical_clustering.py --metric cosine  # Specify distance metric
"""

import sys
from pathlib import Path
from typing import Tuple, List

# Add src directory to path
sys.path.insert(0, str(Path(__file__).parent))

import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.distance import squareform, pdist
from scipy.cluster.hierarchy import dendrogram, linkage
from sklearn.metrics.pairwise import cosine_distances

import config
import utils


def load_language_vectors(languages: List[str]) -> Tuple[np.ndarray, List[str]]:
    """
    Load combined feature vectors for all languages.
    
    Args:
        languages: List of language names to load
    
    Returns:
        Tuple of (matrix of shape (n_languages, n_features), list of language names)
    """
    vectors = []
    loaded_languages = []
    
    for language in languages:
        vectors_dir = utils.get_language_vectors_dir(language)
        combined_file = vectors_dir / "combined.npy"
        
        if not combined_file.exists():
            utils.print_error(f"No combined vector found for {language}")
            continue
        
        vector = np.load(str(combined_file))
        
        if not utils.validate_phone_sequence(vector.tolist(), min_length=1):
            utils.print_error(f"Invalid vector for {language}")
            continue
        
        vectors.append(vector)
        loaded_languages.append(language)
    
    if not vectors:
        return np.array([]).reshape(0, 0), []
    
    matrix = np.array(vectors, dtype=np.float32)
    return matrix, loaded_languages


def compute_distance_matrix(
    vectors: np.ndarray,
    metric: str = "kl_divergence"
) -> np.ndarray:
    """
    Compute pairwise distance matrix between language vectors.
    
    Args:
        vectors: Shape (n_languages, n_features)
        metric: Distance metric ('kl_divergence', 'cosine', 'euclidean')
    
    Returns:
        Distance matrix of shape (n_languages, n_languages)
    """
    n_languages = len(vectors)
    
    if metric == "kl_divergence":
        # KL divergence for normalized probability distributions
        distances = np.zeros((n_languages, n_languages))
        
        for i in range(n_languages):
            for j in range(i + 1, n_languages):
                # Ensure non-negative
                p = np.maximum(vectors[i], 1e-10)
                q = np.maximum(vectors[j], 1e-10)
                
                # Normalize to unit measure if needed
                p = p / np.sum(p)
                q = q / np.sum(q)
                
                # KL(P||Q) = sum(P * log(P / Q))
                kl_div = np.sum(p * np.log(p / q))
                distances[i, j] = kl_div
                distances[j, i] = kl_div
    
    elif metric == "cosine":
        distances = cosine_distances(vectors)
    
    elif metric == "euclidean":
        distances = squareform(pdist(vectors, metric="euclidean"))
    
    else:
        raise ValueError(f"Unknown metric: {metric}")
    
    return distances


def build_dendrogram(
    distance_matrix: np.ndarray,
    language_names: List[str],
    linkage_method: str = "ward",
    metric: str = "kl_divergence",
) -> Tuple[np.ndarray, Path]:
    """
    Build hierarchical clustering dendrogram.
    
    Args:
        distance_matrix: Pairwise distance matrix
        language_names: List of language names
        linkage_method: Linkage method for clustering
        metric: Distance metric name (for title)
    
    Returns:
        Tuple of (linkage_matrix, output_path)
    """
    # Convert to condensed distance matrix for linkage
    condensed_distances = squareform(distance_matrix, checks=False)
    
    # Perform hierarchical clustering
    linkage_matrix = linkage(condensed_distances, method=linkage_method)
    
    # Create dendrogram
    plt.figure(figsize=config.FIGURE_SIZE)
    
    dendro = dendrogram(
        linkage_matrix,
        labels=language_names,
        orientation="top",
        distance_sort="descending",
        show_leaf_counts=False,
        leaf_rotation=45,
    )
    
    # Format title with metric name
    metric_display = {
        "kl_divergence": "KL Divergence",
        "cosine": "Cosine Distance",
        "euclidean": "Euclidean Distance",
    }.get(metric, metric)
    
    plt.title(
        f"Hierarchical Clustering Dendrogram ({metric_display})",
        fontsize=16,
        fontweight="bold"
    )
    plt.xlabel("Language", fontsize=12)
    plt.ylabel(f"{metric_display} Distance", fontsize=12)
    plt.grid(True, alpha=0.3, axis="y")
    
    plt.tight_layout()
    
    # Save plot
    output_dir = config.TREES_DIR
    utils.ensure_directory(output_dir)
    output_path = output_dir / "dendrogram.png"
    plt.savefig(str(output_path), dpi=config.DPI, bbox_inches="tight")
    plt.close()
    
    return linkage_matrix, output_path


def save_clustering_summary(
    distance_matrix: np.ndarray,
    language_names: List[str],
    linkage_matrix: np.ndarray,
    metric: str,
) -> Path:
    """
    Save clustering statistics and summary to file.
    
    Args:
        distance_matrix: Pairwise distance matrix
        language_names: List of language names
        linkage_matrix: Result from scipy.cluster.hierarchy.linkage
        metric: Distance metric name
    
    Returns:
        Path to saved file
    """
    output_dir = config.TREES_DIR
    utils.ensure_directory(output_dir)
    summary_file = output_dir / "clustering_summary.txt"
    
    with open(summary_file, "w") as f:
        f.write("Hierarchical Clustering Summary\n")
        f.write("=" * 70 + "\n\n")
        
        f.write(f"Distance Metric: {metric}\n")
        f.write(f"Number of Languages: {len(language_names)}\n")
        f.write(f"Languages: {', '.join(language_names)}\n\n")
        
        f.write("Distance Matrix Statistics:\n")
        f.write("-" * 70 + "\n")
        f.write(f"  Min distance: {np.min(distance_matrix[distance_matrix > 0]):.6f}\n")
        f.write(f"  Max distance: {np.max(distance_matrix):.6f}\n")
        f.write(f"  Mean distance: {np.mean(distance_matrix[np.triu_indices_from(distance_matrix, k=1)]):.6f}\n")
        f.write(f"  Median distance: {np.median(distance_matrix[np.triu_indices_from(distance_matrix, k=1)]):.6f}\n\n")
        
        f.write("Pairwise Distances (Top 10):\n")
        f.write("-" * 70 + "\n")
        f.write(f"{'Language 1':<20} {'Language 2':<20} {'Distance':<15}\n")
        f.write("-" * 70 + "\n")
        
        # Get pairwise distances
        distances_list = []
        for i in range(len(language_names)):
            for j in range(i + 1, len(language_names)):
                distances_list.append(
                    (language_names[i], language_names[j], distance_matrix[i, j])
                )
        
        # Sort by distance
        distances_list.sort(key=lambda x: x[2])
        
        # Print top 10
        for lang1, lang2, dist in distances_list[:10]:
            f.write(f"{lang1:<20} {lang2:<20} {dist:<15.6f}\n")
        
        f.write("\n" + "=" * 70 + "\n")
        f.write(f"Dendrogram and vectors saved to {config.TREES_DIR}/\n")
    
    return summary_file


def validate_distance_matrix(distance_matrix: np.ndarray) -> bool:
    """Validate that distance matrix has correct properties."""
    # Check symmetric
    if not np.allclose(distance_matrix, distance_matrix.T):
        utils.print_error("Distance matrix is not symmetric")
        return False
    
    # Check non-negative
    if np.any(distance_matrix < 0):
        utils.print_error("Distance matrix contains negative values")
        return False
    
    # Check diagonal is zero
    if not np.allclose(np.diag(distance_matrix), 0):
        utils.print_error("Distance matrix diagonal is not zero")
        return False
    
    return True


def main():
    """Main entry point."""
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Build hierarchical clustering dendrogram from language vectors"
    )
    parser.add_argument(
        "--metric",
        type=str,
        default="kl_divergence",
        choices=["kl_divergence", "cosine", "euclidean"],
        help="Distance metric to use"
    )
    parser.add_argument(
        "--linkage",
        type=str,
        default="ward",
        choices=["ward", "complete", "average", "single"],
        help="Linkage method for hierarchical clustering"
    )
    
    args = parser.parse_args()
    
    utils.print_step(4, "HIERARCHICAL CLUSTERING")
    utils.print_info(f"Distance metric: {args.metric}")
    utils.print_info(f"Linkage method: {args.linkage}")
    
    # Load vectors for all languages
    utils.print_info("Loading combined vectors for all languages...")
    vectors, loaded_languages = load_language_vectors(config.LANGUAGES)
    
    if len(loaded_languages) < 2:
        utils.print_error(f"Need at least 2 languages, found {len(loaded_languages)}")
        return 1
    
    utils.print_success(f"Loaded vectors for {len(loaded_languages)} languages")
    
    # Compute distance matrix
    utils.print_info("Computing distance matrix...")
    distance_matrix = compute_distance_matrix(vectors, metric=args.metric)
    
    # Validate
    if not validate_distance_matrix(distance_matrix):
        return 1
    
    utils.print_success(
        f"Distance matrix computed: shape {distance_matrix.shape}, "
        f"max distance {distance_matrix.max():.6f}"
    )
    
    # Build dendrogram
    utils.print_info("Building dendrogram...")
    linkage_matrix, dendrogram_path = build_dendrogram(
        distance_matrix,
        loaded_languages,
        linkage_method=args.linkage,
        metric=args.metric,
    )
    
    utils.print_success(f"Dendrogram saved to {dendrogram_path}")
    
    # Save summary
    summary_path = save_clustering_summary(
        distance_matrix,
        loaded_languages,
        linkage_matrix,
        args.metric,
    )
    
    utils.print_success(f"Summary saved to {summary_path}")
    
    utils.print_summary("CLUSTERING SUMMARY", {
        "Languages": len(loaded_languages),
        "Distance metric": args.metric,
        "Linkage method": args.linkage,
        "Max distance": f"{distance_matrix.max():.6f}",
        "Output directory": str(config.TREES_DIR),
    })
    
    utils.print_success("Hierarchical clustering complete!")
    return 0


if __name__ == "__main__":
    exit(main())
