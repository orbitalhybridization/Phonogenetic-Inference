#!/usr/bin/env python3
"""
One-Click Pipeline Runner
==========================
Simple way to run the entire pipeline at once.

Usage:
    python run_pipeline.py
    
No arguments needed - the script will process all WAV files in input/ automatically.
"""

import sys
import os
from pathlib import Path
import subprocess

# Get the script directory
SCRIPT_DIR = Path(__file__).parent
INPUT_DIR = SCRIPT_DIR / "input"
SRC_DIR = SCRIPT_DIR / "src"

# Colors for terminal output
GREEN = "\033[92m"
BLUE = "\033[94m"
RED = "\033[91m"
YELLOW = "\033[93m"
BOLD = "\033[1m"
RESET = "\033[0m"


def print_header(text):
    """Print a formatted header."""
    print(f"\n{BOLD}{BLUE}{'=' * 76}{RESET}")
    print(f"{BOLD}{BLUE}{text:^76}{RESET}")
    print(f"{BOLD}{BLUE}{'=' * 76}{RESET}\n")


def print_success(text):
    """Print success message."""
    print(f"{GREEN}[OK]{RESET}  {text}")


def print_error(text):
    """Print error message."""
    print(f"{RED}[ERROR]{RESET}  {text}")


def print_step(step_num, text):
    """Print step header."""
    print(f"\n{BOLD}━{'━' * 74}━{RESET}")
    print(f"{BOLD}STEP {step_num}: {text}{RESET}")
    print(f"{BOLD}━{'━' * 74}━{RESET}\n")


def check_input_files():
    """Check if there are WAV files in input directory."""
    wav_files = list(INPUT_DIR.glob("**/*.wav"))
    
    if not wav_files:
        print_error(f"No WAV files found in input/ directory")
        print(f"\n{YELLOW}Please organize your audio files like this:{RESET}\n")
        print("  input/english/")
        print("  ├── audio1.wav")
        print("  ├── audio2.wav")
        print("  └── ...")
        print("")
        print("  input/french/")
        print("  ├── audio1.wav")
        print("  └── ...")
        print("")
        return False
    
    # Count by language folder
    languages = {}
    for wav_file in wav_files:
        lang = wav_file.parent.name
        languages[lang] = languages.get(lang, 0) + 1
    
    print_success(f"Found {len(wav_files)} WAV files:")
    for lang, count in sorted(languages.items()):
        print(f"  {lang}: {count} files")
    print()
    
    return True


def run_script(script_num, script_name, description):
    """Run a pipeline script."""
    print_step(script_num, description)
    
    script_path = SRC_DIR / script_name
    
    if not script_path.exists():
        print_error(f"Script not found: {script_path}")
        return False
    
    try:
        result = subprocess.run([sys.executable, str(script_path)], check=True)
        print()
        return True
    except subprocess.CalledProcessError as e:
        print_error(f"Script failed with exit code {e.returncode}")
        return False
    except Exception as e:
        print_error(f"Error running script: {e}")
        return False


def print_results():
    """Print information about output files."""
    print_header("PIPELINE COMPLETE")
    print(f"{GREEN}[OK]{RESET}  All steps finished successfully!\n")
    
    print(f"{BOLD}Your results are in the output/ folder:{RESET}\n")
    
    print(f"{YELLOW}Phone Analysis:{RESET}")
    print("  output/phone_summaries/[language]/")
    print("    - stats.txt (frequency statistics)")
    print("    - top_phones.png (top 10 chart)")
    print("    - phone_distribution.png (pie chart)\n")
    
    print(f"{YELLOW}Feature Vectors:{RESET}")
    print("  output/vectors/[language]/")
    print("    - combined.npy (unified feature vector)")
    print("    - phones.npy, 2grams.npy, 3grams.npy, 4grams.npy")
    print("    - feature_stats.txt (dimension breakdown)\n")
    
    print(f"{YELLOW}Phylogenetic Tree:{RESET}")
    print("  output/results/trees/")
    print("    - dendrogram.png (language relationships)")
    print("    - clustering_summary.txt (distance statistics)\n")
    
    dendrogram_path = Path("output/results/trees/dendrogram.png")
    if dendrogram_path.exists():
        print(f"{GREEN}[OK]{RESET}  Open {BOLD}output/results/trees/dendrogram.png{RESET} to see your language tree\n")


def main():
    """Main entry point."""
    print_header("PHONOGENETIC INFERENCE PIPELINE - QUICK START")
    
    # Check input files
    if not check_input_files():
        sys.exit(1)
    
    # Run all steps
    steps = [
        (1, "1_phone_recognition.py", "Recognizing phones from audio files"),
        (2, "2_phone_summarization.py", "Summarizing phone statistics per language"),
        (3, "3_vectorization.py", "Creating feature vectors (phones + n-grams)"),
        (4, "4_hierarchical_clustering.py", "Building phylogenetic dendrogram"),
    ]
    
    for step_num, script_name, description in steps:
        if not run_script(step_num, script_name, description):
            print_error(f"Pipeline failed at step {step_num}")
            sys.exit(1)
    
    # Success!
    print_results()


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print(f"\n{RED}Pipeline interrupted by user{RESET}\n")
        sys.exit(1)
