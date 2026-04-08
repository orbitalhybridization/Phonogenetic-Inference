#!/usr/bin/env python3
"""
Script 1: Phone Recognition
============================
Recognize phones from audio files using Allosaurus.

Input: WAV files organized in input/[language]/
Output: Recognized phone sequences in output/phones/[language]/[filename].npy

Usage:
    python 1_phone_recognition.py              # Process all languages
    python 1_phone_recognition.py --lang garifuna  # Process specific language
"""

import sys
from pathlib import Path
from typing import List, Tuple

# Add src directory to path
sys.path.insert(0, str(Path(__file__).parent))

import numpy as np
from tqdm import tqdm
from allosaurus.app import read_recognizer

import config
import utils


def find_wav_files(language: str) -> List[Path]:
    """
    Find all WAV files for a given language.
    
    Args:
        language: Language name
    
    Returns:
        List of Path objects for WAV files
    """
    lang_input_dir = utils.get_language_input_dir(language)
    
    if not lang_input_dir.exists():
        return []
    
    wav_files = sorted(lang_input_dir.glob("*.wav"))
    return wav_files


def recognize_phones_from_file(wav_file: Path, recognizer) -> Tuple[List[str], bool]:
    """
    Recognize phones from a single WAV file.
    
    Args:
        wav_file: Path to WAV file
        recognizer: Allosaurus recognizer object
    
    Returns:
        Tuple of (phones list, success flag)
    """
    try:
        # Use recognize() method (not recognize_custom)
        phones = recognizer.recognize(str(wav_file))
        
        # Convert token string to list if needed
        if isinstance(phones, str):
            phones = phones.split()
        else:
            phones = list(phones)
        
        # Validate non-empty
        if not phones:
            return [], False
        
        return phones, True
    
    except Exception as e:
        utils.print_error(f"Recognition failed for {wav_file.name}: {e}")
        return [], False


def process_language(language: str, recognizer) -> Tuple[int, int, int]:
    """
    Process all WAV files for a language.
    
    Args:
        language: Language name
        recognizer: Allosaurus recognizer object
    
    Returns:
        Tuple of (total_files, successful, total_phones)
    """
    wav_files = find_wav_files(language)
    
    if not wav_files:
        utils.print_error(f"No WAV files found in {config.INPUT_DIR / language}")
        return 0, 0, 0
    
    successful = 0
    total_phones = 0
    
    for wav_file in tqdm(wav_files, desc=f"Processing {language}", unit="file"):
        phones, success = recognize_phones_from_file(wav_file, recognizer)
        
        if success:
            # Save to output
            utils.save_phone_sequence(phones, language, wav_file.name)
            successful += 1
            total_phones += len(phones)
    
    return len(wav_files), successful, total_phones


def main():
    """Main entry point."""
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Recognize phones from audio files using Allosaurus"
    )
    parser.add_argument(
        "--lang",
        type=str,
        default=None,
        help=f"Process specific language (default: all). Options: {', '.join(config.LANGUAGES)}"
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
    
    utils.print_step(1, "PHONE RECOGNITION")
    utils.print_info(f"Languages to process: {', '.join(languages)}")
    utils.print_info("Loading Allosaurus recognizer...")
    
    try:
        recognizer = read_recognizer()
        utils.print_success("Recognizer loaded")
    except Exception as e:
        utils.print_error(f"Failed to load recognizer: {e}")
        return 1
    
    # Process each language
    total_summary = {}
    
    for language in languages:
        total_files, successful, total_phones = process_language(language, recognizer)
        
        if total_files == 0:
            continue
        
        total_summary[language] = {
            "files_processed": successful,
            "total_files": total_files,
            "total_phones": total_phones,
        }
        
        if successful > 0:
            utils.print_success(
                f"{language}: Processed {successful}/{total_files} WAVs, "
                f"recognized {total_phones} phones"
            )
        else:
            utils.print_error(f"{language}: Failed to process any files")
    
    # Print summary
    if total_summary:
        utils.print_summary("RECOGNITION SUMMARY", {
            k: f"{v['files_processed']}/{v['total_files']} files, {v['total_phones']} phones"
            for k, v in total_summary.items()
        })
        utils.print_success("Phone recognition complete!")
        return 0
    else:
        utils.print_error("No files processed")
        return 1


if __name__ == "__main__":
    exit(main())
