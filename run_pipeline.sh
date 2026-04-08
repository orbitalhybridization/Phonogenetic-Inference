#!/bin/bash
# Simple one-click pipeline runner for non-technical users
# Usage: ./run_pipeline.sh

set -e  # Exit on any error

echo "========================================================================"
echo "PHONOGENETIC INFERENCE PIPELINE - QUICK START"
echo "========================================================================"
echo ""

# Check if input folder has any WAV files
if ! find input -name "*.wav" -type f -quit 2>/dev/null | grep -q .; then
    echo "ERROR: No WAV files found in input/ directory"
    echo ""
    echo "Please add WAV files organized by language:"
    echo "  input/english/*.wav"
    echo "  input/french/*.wav"
    echo "  input/spanish/*.wav"
    echo "  etc."
    echo ""
    exit 1
fi

TOTAL_WAVS=$(find input -name "*.wav" -type f | wc -l)
echo "Found $TOTAL_WAVS WAV files - starting pipeline..."
echo ""

# Make scripts executable
chmod +x src/*.py

# Step 1: Phone Recognition
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "STEP 1: Recognizing phones from audio (this may take a few minutes)..."
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
python src/1_phone_recognition.py
echo ""

# Step 2: Phone Summarization
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "STEP 2: Summarizing phone statistics per language..."
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
python src/2_phone_summarization.py
echo ""

# Step 3: Vectorization
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "STEP 3: Creating feature vectors (phones + n-grams)..."
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
python src/3_vectorization.py
echo ""

# Step 4: Hierarchical Clustering
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "STEP 4: Building phylogenetic dendrogram..."
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
python src/4_hierarchical_clustering.py
echo ""

# Success message
echo "========================================================================"
echo "✅ PIPELINE COMPLETE!"
echo "========================================================================"
echo ""
echo "Your results are ready in the output/ folder:"
echo ""
echo "Phone Analysis:"
echo "  output/phone_summaries/[language]/"
echo "    - stats.txt (frequency table)"
echo "    - top_phones.png (top 10 chart)"
echo "    - phone_distribution.png (pie chart)"
echo ""
echo "Feature Vectors:"
echo "  output/vectors/[language]/"
echo "    - combined.npy (unified vector for all languages)"
echo "    - phones.npy, 2grams.npy, 3grams.npy, 4grams.npy"
echo "    - feature_stats.txt (dimension breakdown)"
echo ""
echo "Phylogenetic Tree:"
echo "  output/results/trees/"
echo "    - dendrogram.png (language relationships)"
echo "    - clustering_summary.txt (distance statistics)"
echo ""
echo "Open output/results/trees/dendrogram.png to see your language tree."
echo ""
