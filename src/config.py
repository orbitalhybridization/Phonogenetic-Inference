"""
Central configuration for phono_phylo_inference pipeline.
Edit these settings to customize behavior across all scripts.
"""

import os
from pathlib import Path

# ============================================================================
# PATHS
# ============================================================================
PROJECT_ROOT = Path(__file__).parent.parent
INPUT_DIR = PROJECT_ROOT / "input"
OUTPUT_DIR = PROJECT_ROOT / "output"

PHONES_DIR = OUTPUT_DIR / "phones"
PHONE_SUMMARIES_DIR = OUTPUT_DIR / "phone_summaries"
VECTORS_DIR = OUTPUT_DIR / "vectors"
RESULTS_DIR = OUTPUT_DIR / "results"
TREES_DIR = RESULTS_DIR / "trees"

# Create output directories if they don't exist
for dir_path in [PHONES_DIR, PHONE_SUMMARIES_DIR, VECTORS_DIR, TREES_DIR]:
    dir_path.mkdir(parents=True, exist_ok=True)

# ============================================================================
# LANGUAGE CONFIGURATION
# ============================================================================
# List of languages to process. Easy to add/remove without editing scripts.
LANGUAGES = [
    "english",
    "french",
    "garifuna",
    "parecis",
    "piapoco",
    "portuguese",
    "spanish",
    "wayuu",
]

# ============================================================================
# PHONE RECOGNITION SETTINGS
# ============================================================================
AUDIO_EXTENSION = ".wav"

# ============================================================================
# N-GRAM SETTINGS
# ============================================================================
# Range of n-grams to extract (e.g., 2 to 4 means bigrams, trigrams, 4-grams)
NGRAM_MIN = 2
NGRAM_MAX = 4
NGRAM_SIZES = list(range(NGRAM_MIN, NGRAM_MAX + 1))  # [2, 3, 4]

# ============================================================================
# VECTORIZATION SETTINGS
# ============================================================================
# Vector normalization method: 'l2' or None
VECTOR_NORMALIZATION = "l2"

# Vocabulary building across languages: 'union' or 'intersection'
# - 'union': all unique phones/ngrams across all languages (default, captures full diversity)
# - 'intersection': only phones/ngrams common to all languages (for conservative analysis)
VOCABULARY_MODE = "union"

# ============================================================================
# CLUSTERING SETTINGS
# ============================================================================
# Distance metric for dendrogram: 'kl_divergence', 'cosine', 'euclidean'
DISTANCE_METRIC = "kl_divergence"

# Linkage method for hierarchical clustering
LINKAGE_METHOD = "ward"

# ============================================================================
# VISUALIZATION SETTINGS
# ============================================================================
DPI = 300
FIGURE_SIZE = (12, 8)
CMAP = "viridis"  # Colormap for visualizations

# ============================================================================
# LOGGING & VERBOSITY
# ============================================================================
VERBOSE = True  # Print detailed progress messages
