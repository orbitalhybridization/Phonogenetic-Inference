#!/usr/bin/env python3
"""
Script 2: Phone Summarization
==============================
Aggregate phone statistics and create visualizations per language.

Input: Recognized phones from output/phones/[language]/
Output: Statistics and visualizations in output/phone_summaries/[language]/

Usage:
    python 2_phone_summarization.py              # Summarize all languages
    python 2_phone_summarization.py --lang garifuna  # Summarize specific language
"""

import sys
from pathlib import Path
from typing import Dict, Tuple

# Add src directory to path
sys.path.insert(0, str(Path(__file__).parent))

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter

import config
import utils


def generate_statistics(phones: list) -> Dict:
    """
    Generate frequency and statistical information about phones.
    
    Args:
        phones: List of phone symbols
    
    Returns:
        Dictionary with statistics
    """
    if not phones:
        return {}
    
    counter = Counter(phones)
    total = len(phones)
    unique_count = len(counter)
    
    # Top 10 phones
    top_10 = counter.most_common(10)
    
    # Create distributions
    frequency_dist = {phone: count / total for phone, count in counter.items()}
    
    return {
        "total_phones": total,
        "unique_phones": unique_count,
        "counter": counter,
        "frequency_dist": frequency_dist,
        "top_10": top_10,
    }


def save_statistics_file(language: str, stats: Dict) -> Path:
    """
    Save statistics to a text file.
    
    Args:
        language: Language name
        stats: Statistics dictionary
    
    Returns:
        Path to saved file
    """
    output_dir = utils.get_language_summaries_dir(language)
    stats_file = output_dir / "stats.txt"
    
    with open(stats_file, "w") as f:
        f.write(f"Phone Statistics for {language}\n")
        f.write("=" * 70 + "\n\n")
        
        f.write(f"Total phones: {stats['total_phones']}\n")
        f.write(f"Unique phones: {stats['unique_phones']}\n\n")
        
        f.write("Top 10 Most Common Phones:\n")
        f.write("-" * 70 + "\n")
        for i, (phone, count) in enumerate(stats['top_10'], 1):
            freq = count / stats['total_phones'] * 100
            f.write(f"  {i:2d}. {phone:15s}  {count:6d} occurrences  ({freq:5.2f}%)\n")
        
        f.write("\n" + "=" * 70 + "\n")
    
    return stats_file


def plot_top_phones(language: str, stats: Dict) -> Path:
    """
    Create bar chart of top 10 phones.
    
    Args:
        language: Language name
        stats: Statistics dictionary
    
    Returns:
        Path to saved plot
    """
    output_dir = utils.get_language_summaries_dir(language)
    plot_file = output_dir / "top_phones.png"
    
    phones, counts = zip(*stats['top_10'])
    
    plt.figure(figsize=config.FIGURE_SIZE)
    bars = plt.barh(range(len(phones)), counts, color="steelblue", alpha=0.8)
    plt.yticks(range(len(phones)), phones)
    plt.xlabel("Frequency", fontsize=12)
    plt.ylabel("Phone", fontsize=12)
    plt.title(f"Top 10 Most Common Phones - {language.capitalize()}", fontsize=14, fontweight="bold")
    plt.gca().invert_yaxis()  # Most common at top
    
    # Add count labels on bars
    for i, (bar, count) in enumerate(zip(bars, counts)):
        plt.text(count, i, f" {count}", va="center", fontsize=10)
    
    plt.tight_layout()
    plt.savefig(plot_file, dpi=config.DPI)
    plt.close()
    
    return plot_file


def plot_phone_distribution_pie(language: str, stats: Dict, top_n: int = 10) -> Path:
    """
    Create pie chart of phone distribution.
    
    Args:
        language: Language name
        stats: Statistics dictionary
        top_n: Number of top phones to show (others grouped as "Other")
    
    Returns:
        Path to saved plot
    """
    output_dir = utils.get_language_summaries_dir(language)
    plot_file = output_dir / "phone_distribution.png"
    
    # Get top N and aggregate the rest
    phone_counts = dict(stats['counter'])
    sorted_phones = sorted(phone_counts.items(), key=lambda x: x[1], reverse=True)
    
    labels = [phone for phone, _ in sorted_phones[:top_n]]
    sizes = [count for _, count in sorted_phones[:top_n]]
    
    # Add "Other" if there are more phones
    if len(sorted_phones) > top_n:
        other_size = sum(count for _, count in sorted_phones[top_n:])
        labels.append("Other")
        sizes.append(other_size)
    
    plt.figure(figsize=(10, 8))
    colors = sns.color_palette("Set2", len(labels))
    wedges, texts, autotexts = plt.pie(
        sizes,
        labels=labels,
        autopct="%1.1f%%",
        colors=colors,
        startangle=90,
        textprops={"fontsize": 10}
    )
    
    plt.title(f"Phone Distribution - {language.capitalize()}", fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.savefig(plot_file, dpi=config.DPI)
    plt.close()
    
    return plot_file


def summarize_language(language: str) -> Tuple[int, int, int]:
    """
    Generate summary for a single language.
    
    Args:
        language: Language name
    
    Returns:
        Tuple of (phones_processed, unique_phones, total_phones)
    """
    # Load all phone sequences for the language
    all_phones = utils.aggregate_phone_sequences(language)
    
    if not all_phones:
        utils.print_error(f"No phone sequences found for {language}")
        return 0, 0, 0
    
    # Generate statistics
    stats = generate_statistics(all_phones)
    
    # Save statistics file
    save_statistics_file(language, stats)
    
    # Create visualizations
    plot_top_phones(language, stats)
    plot_phone_distribution_pie(language, stats)
    
    return len(all_phones), stats['unique_phones'], stats['total_phones']


def main():
    """Main entry point."""
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Summarize phone statistics per language"
    )
    parser.add_argument(
        "--lang",
        type=str,
        default=None,
        help=f"Summarize specific language (default: all). Options: {', '.join(config.LANGUAGES)}"
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
    
    utils.print_step(2, "PHONE SUMMARIZATION")
    utils.print_info(f"Languages to process: {', '.join(languages)}")
    
    summary_data = {}
    
    for language in languages:
        phones_processed, unique_count, total_count = summarize_language(language)
        
        if total_count == 0:
            utils.print_error(f"{language}: No data to summarize")
            continue
        
        summary_data[language] = {
            "total_phones": total_count,
            "unique_phones": unique_count,
        }
        
        utils.print_success(
            f"{language}: Summarized {total_count} phones "
            f"({unique_count} unique)"
        )
    
    # Print summary
    if summary_data:
        utils.print_summary("SUMMARIZATION SUMMARY", {
            k: f"{v['total_phones']} total, {v['unique_phones']} unique"
            for k, v in summary_data.items()
        })
        utils.print_success("Phone summarization complete!")
        return 0
    else:
        utils.print_error("No languages summarized")
        return 1


if __name__ == "__main__":
    exit(main())
