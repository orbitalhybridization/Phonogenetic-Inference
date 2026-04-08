# Phono-Phylo Inference Pipeline

A modular, user-friendly pipeline for analyzing phonetic diversity across languages. Recognizes phones from audio files, computes linguistic features, and infers phylogenetic relationships through hierarchical clustering.

## Quick Start

```bash
# 1. Place audio files in input/ folder
#    input/english/*.wav
#    input/french/*.wav
#    etc.

# 2. Run one of these commands:

# Option A: Shell script (Mac/Linux):
./run_pipeline.sh

# Option B: Python (Any OS):
python run_pipeline.py
```

Pipeline will:
- Recognize phones from all audio files
- Summarize statistics per language
- Create feature vectors
- Build a phylogenetic tree (dendrogram)

Results go to the `output/` folder with visualizations included.

---

## Quick Start (Manual Steps)

### 1. **Setup** (One-time)

```bash
# Clone repository and install dependencies
pip install -r requirements.txt
```

### 2. **Prepare Input**

Create language-named folders in `input/` and place your WAV files:

```
input/
├── english/
│   ├── sample1.wav
│   ├── sample2.wav
│   └── ...
├── french/
│   ├── audio1.wav
│   └── ...
├── garifuna/
│   ├── file1.wav
│   └── ...
└── [language_name]/
    └── *.wav
```

### 3. **Run the Pipeline** (Sequential steps)

```bash
# Step 1: Recognize phones from audio
python src/1_phone_recognition.py

# Step 2: Summarize phone statistics per language
python src/2_phone_summarization.py

# Step 3: Create feature vectors (phones + n-grams)
python src/3_vectorization.py

# Step 4: Build hierarchical clustering dendrogram
python src/4_hierarchical_clustering.py
```

### 4. **View Results**

Results are organized in `output/`:

```
output/
├── phones/[language]/          # Recognized phones per WAV file
├── phone_summaries/[language]/ # Statistics and visualizations
├── vectors/[language]/         # Feature vectors
└── results/
    ├── trees/                  # Dendrogram and clustering analysis
    └── [language]/             # Per-language visualizations
```

---

## Detailed Usage

### One-Script Convenience Options

If you prefer not to run 4 separate scripts, use one of these all-in-one runners:

**Option 1: Shell script (Mac/Linux)**
```bash
chmod +x run_pipeline.sh
./run_pipeline.sh
```

**Option 2: Python script (Windows/Mac/Linux)**
```bash
python run_pipeline.py
```

Both do the same thing: automatically run all 4 steps in sequence with clear progress messages.

---

### Script 1: Phone Recognition

Recognizes phones from audio files using the [Allosaurus](https://github.com/xinjli/allosaurus) model.

```bash
python src/1_phone_recognition.py [OPTIONS]
```

**Options:**
- `--lang LANGUAGE` - Process specific language (default: all)
- Example: `python src/1_phone_recognition.py --lang garifuna`

**Input:**
- WAV files in `input/[language]/`

**Output:**
- `output/phones/[language]/[filename].npy` - Phone sequences as numpy arrays

**What it does:**
- Loads the Allosaurus recognizer (automatic model download)
- Processes each WAV file to extract phone sequences
- Validates output and shows progress
- Prints summary of phones recognized per language

---

### Script 2: Phone Summarization

Aggregates phone recognition results and generates statistics and visualizations.

```bash
python src/2_phone_summarization.py [OPTIONS]
```

**Options:**
- `--lang LANGUAGE` - Summarize specific language (default: all)

**Input:**
- Phone sequences from `output/phones/[language]/`

**Output:**
- `output/phone_summaries/[language]/stats.txt` - Frequency statistics
- `output/phone_summaries/[language]/top_phones.png` - Bar chart of top 10 phones
- `output/phone_summaries/[language]/phone_distribution.png` - Distribution pie chart

**What it does:**
- Aggregates all phones from all WAV files per language
- Computes frequency statistics
- Identifies top 10 most common phones
- Creates visualizations of phone inventory and distribution

---

### Script 3: Vectorization

Creates rich feature vectors combining phones and n-grams.

```bash
python src/3_vectorization.py [OPTIONS]
```

**Options:**
- `--lang LANGUAGE` - Vectorize specific language (default: all)

**Input:**
- Phone sequences from `output/phones/[language]/`

**Output:**
- `output/vectors/[language]/phones.npy` - Phone frequency vector
- `output/vectors/[language]/2grams.npy` - 2-gram frequency vector
- `output/vectors/[language]/3grams.npy` - 3-gram frequency vector
- `output/vectors/[language]/4grams.npy` - 4-gram frequency vector
- `output/vectors/[language]/combined.npy` - Concatenated feature vector (used for clustering)
- `output/vectors/[language]/feature_stats.txt` - Vector dimension statistics

**What it does:**
- Extracts phone bigrams, trigrams, and 4-grams from sequences
- Normalizes frequencies by total phone count
- L2-normalizes each feature type separately
- Applies equal weighting (0.25 each) to phones and each n-gram type
- Creates combined feature vector for downstream analysis

**Vector Details:**
- Each feature type is L2-normalized independently
- Equal weighting: 25% phones + 25% 2-grams + 25% 3-grams + 25% 4-grams
- Combined vector captures both inventory and sequential patterns
- Suitable for distance metrics (KL divergence, cosine, etc.)

---

### Script 4: Hierarchical Clustering

Builds phylogenetic trees using hierarchical clustering.

```bash
python src/4_hierarchical_clustering.py [OPTIONS]
```

**Options:**
- `--metric METRIC` - Distance metric (default: `kl_divergence`)
  - Options: `kl_divergence`, `cosine`, `euclidean`
- `--linkage METHOD` - Linkage method (default: `ward`)
  - Options: `ward`, `complete`, `average`, `single`
- Examples:
  ```bash
  python src/4_hierarchical_clustering.py --metric cosine
  python src/4_hierarchical_clustering.py --linkage complete
  ```

**Input:**
- Combined feature vectors from `output/vectors/[language]/combined.npy`

**Output:**
- `output/results/trees/dendrogram.png` - Hierarchical clustering dendrogram
- `output/results/trees/clustering_summary.txt` - Distance statistics and language pairs

**What it does:**
- Loads combined vectors for all languages
- Computes pairwise distance matrix using specified metric
- Performs hierarchical clustering with specified linkage method
- Generates publication-ready dendrogram visualization
- Computes and reports distance statistics

**Distance Metrics:**
- **KL Divergence** (default): Measures information divergence between probability distributions. Good for normalized frequency vectors.
- **Cosine Distance**: Measures angle between vectors. Invariant to magnitude, focuses on direction.
- **Euclidean Distance**: Straight-line distance in feature space.

---

## Configuration

Edit `src/config.py` to customize pipeline settings without modifying scripts:

```python
# Languages to process
LANGUAGES = ["english", "french", "garifuna", "parecis", ...]

# N-gram range
NGRAM_MIN = 2
NGRAM_MAX = 4

# Distance metric for clustering
DISTANCE_METRIC = "kl_divergence"

# Linkage method
LINKAGE_METHOD = "ward"

# Output directories (created automatically)
INPUT_DIR = PROJECT_ROOT / "input"
OUTPUT_DIR = PROJECT_ROOT / "output"
```

---

## Output Structure

```
output/
├── phones/
│   ├── english/
│   │   ├── audio1.npy          # Phone sequence
│   │   ├── audio2.npy
│   │   └── ...
│   └── [language]/
│
├── phone_summaries/
│   ├── english/
│   │   ├── stats.txt            # Frequency statistics
│   │   ├── top_phones.png       # Top 10 distribution
│   │   └── phone_distribution.png  # Pie chart
│   └── [language]/
│
├── vectors/
│   ├── english/
│   │   ├── phones.npy           # (n_unique_phones,)
│   │   ├── 2grams.npy           # (n_unique_2grams,)
│   │   ├── 3grams.npy           # (n_unique_3grams,)
│   │   ├── 4grams.npy           # (n_unique_4grams,)
│   │   ├── combined.npy         # (total_features,)
│   │   └── feature_stats.txt    # Dimension information
│   └── [language]/
│
└── results/
    └── trees/
        ├── dendrogram.png       # Main hierarchical clustering tree
        └── clustering_summary.txt  # Distance statistics
```

---

## Requirements

- **Python 3.7+**
- **Dependencies** (see `requirements.txt`):
  - `numpy` - Array operations
  - `scipy` - Distance and clustering algorithms
  - `matplotlib` - Visualization
  - `seaborn` - Enhanced plotting
  - `scikit-learn` - Preprocessing and distances
  - `allosaurus` - Phone recognition from audio
  - `networkx` - Graph utilities

Install all at once:

```bash
pip install -r requirements.txt
```

---

## Troubleshooting

### Q: Allosaurus fails to download model
**A:** First run will download the model (~300MB). Ensure internet connection and sufficient disk space.

```bash
# Test if Allosaurus works
python -c "from allosaurus.app import read_recognizer; r = read_recognizer()"
```

### Q: No "combined.npy" vectors for a language
**A:** Make sure you've run steps 1 and 3 successfully. Check:

```bash
ls output/phones/[language]/*.npy     # Step 1 output
ls output/vectors/[language]/         # Step 3 output
```

### Q: Dendrogram script says "Need at least 2 languages"
**A:** Vectorize at least 2 languages before clustering:

```bash
python src/3_vectorization.py          # Run for all available languages
python src/4_hierarchical_clustering.py  # Then cluster
```

### Q: How do I process only one language?
**A:** All scripts support `--lang` flag:

```bash
python src/1_phone_recognition.py --lang english
python src/2_phone_summarization.py --lang english
python src/3_vectorization.py --lang english
```

### Q: Can I use different distance metrics?
**A:** Yes! Script 4 supports multiple metrics:

```bash
python src/4_hierarchical_clustering.py --metric cosine
python src/4_hierarchical_clustering.py --metric euclidean
```

---

## Understanding the Vectors

### Phone Vector
- **Dimension**: Number of unique phones across the language's inventory
- **Values**: Frequency of each phone normalized by total phones
- **Example**: `[0.15, 0.12, 0.08, ...]` for 50 unique phones

### N-gram Vectors
- **2-gram**: All unique pairs of consecutive phones
- **3-gram**: All unique triples of consecutive phones
- **4-gram**: All unique quadruples of consecutive phones
- **Values**: Frequency normalized by total phones

### Combined Vector
- **Composition**: `[phone_vector(0.25), 2gram_vector(0.25), 3gram_vector(0.25), 4gram_vector(0.25)]`
- **Total dimensions**: Sum of all unique phones + all unique n-grams
- **Normalized**: L2 normalized for use with distance metrics

---

## Citation

If you use this pipeline in research, please cite the Allosaurus model:

> Li, X. I. (2021). The Allosaurus Model: Advances in Language-independent Speech Recognition.

And cite this repository (once published).

---

## License

[Your License Here]

---

## Contact & Questions

For issues, questions, or feature requests, please open an issue on GitHub.

---

## Appendix: Step-by-Step Example

```bash
# Set up a test with 2 languages
mkdir -p input/english input/french

# Copy sample WAV files
cp /path/to/english_samples/*.wav input/english/
cp /path/to/french_samples/*.wav input/french/

# Step 1: Recognize phones
python src/1_phone_recognition.py
# Output: output/phones/english/*.npy, output/phones/french/*.npy

# Step 2: Summarize
python src/2_phone_summarization.py
# Output: output/phone_summaries/{english,french}/

# Step 3: Vectorize
python src/3_vectorization.py
# Output: output/vectors/{english,french}/*.npy

# Step 4: Cluster
python src/4_hierarchical_clustering.py
# Output: output/results/trees/dendrogram.png

# View the dendrogram
open output/results/trees/dendrogram.png
```

---

## Architecture Overview

```
input/[language]/*.wav
        ↓
[1_phone_recognition.py]
        ↓
output/phones/[language]/*.npy
        ↓
[2_phone_summarization.py]
        ↓
output/phone_summaries/[language]/{stats.txt, *.png}
        ↓
[3_vectorization.py]
        ↓
output/vectors/[language]/{phones, 2grams, 3grams, 4grams, combined}.npy
        ↓
[4_hierarchical_clustering.py]
        ↓
output/results/trees/{dendrogram.png, clustering_summary.txt}
```

Each script is independent and can be rerun without affecting others.
