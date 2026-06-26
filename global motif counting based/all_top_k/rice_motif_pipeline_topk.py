# -*- coding: utf-8 -*-
#!/usr/bin/env python3
"""
Multi-Variety Global Motif Analysis Pipeline for Rice Drought Response (Top-K Version)
=======================================================================================
This pipeline filters data for top-K varieties (by sample count) and processes motifs.

Usage:
  python rice_motif_pipeline_topk.py --top_k 3
  python rice_motif_pipeline_topk.py --top_k 5

Arguments:
  --top_k: Number of top varieties (by sample count) to include (1-13)
"""

import os
import re
import sys
import argparse
import warnings
from typing import Dict, Any, Tuple, List

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

warnings.filterwarnings("ignore")


# =============================================================================
# CONFIGURATION
# =============================================================================

INPUT_FILE = "/home/saiful/DeepRice/motif_based_classification_ALL_DATA/data_preprocess/RiceMetaSys_Drought_merged_all_excels_5.csv"
BASE_OUTPUT_DIR = "/home/saiful/DeepRice_Task1/motif_based_classification_ALL_DATA/MSc_thesis2/global motif counting based/all_top_k"
RANDOM_SEED = 42


# =============================================================================
# MOTIF DEFINITIONS (regex patterns)
# =============================================================================

MOTIFS = {
    # ABA-responsive elements
    'ABRE': 'ACGTG[GT][CT]',
    'ABRE_core': 'ACGTG',
    'CE1': 'TGCCACCGG',
    'CE3': 'ACGCGTGTC',
    
    # Dehydration-responsive elements
    'DRE_CRT': 'CCGAC',
    'DRE_full': '[AG]CCGAC',
    'LTRE': 'CCGAC',
    
    # MYB binding sites
    'MYB_core': 'C[ACGT]GTT[AG]',
    'MYB1AT': '[AT]AACCA',
    'MYB2': 'TAACTG',
    'MYBCORE': 'C[ACGT]GTT[AG]',
    'MYC': 'CA[ACGT][ACGT]TG',
    
    # W-box (WRKY binding)
    'WBOX': 'TTGAC[CT]',
    'WBOX_core': 'TGAC',
    
    # GT elements
    'GT1': 'G[AG][AT]AA[AT]',
    'GT1_core': 'GAAAAA',
    
    # Heat shock elements
    'HSE': 'GAA[ACGT][ACGT]TTC',
    'HSE_core': '[ACGT]GAA[ACGT]',
    
    # Ethylene responsive
    'GCC_box': 'GCCGCC',
    'ERE': 'A[AT]TTCAAA',
    
    # Light/circadian elements
    'GBOX': 'CACGTG',
    'IBOX': 'GATAAG',
    'CAAT_box': 'CCAAT',
    
    # Stress responsive
    'STRE': 'AGGGG',
    'LTRE_like': 'ACCGACA',
    
    # Wound/pathogen response
    'JASE': 'CGTCA',
    'TBOX': 'ACTTTG',
    
    # NAC binding sites
    'NAC_core': 'CACG',
    'NACRS': 'CATGTG',
    
    # bZIP binding sites
    'ACGT_core': 'ACGT',
    'ABRE_like': '[AC]ACGT[GC][CGT]',
    
    # Auxin responsive
    'AuxRE': 'TGTCTC',
    'AuxRE_rev': 'GAGACA',
    
    # Root specific
    'ROOTMOTIF': 'ATATT',
    
    # Additional elements
    'AS1': 'TGACG',
    'TATA_box': 'TATAAA',
    'CACTFTPPCA1': '[CT]ACT',
    'ICE_box': 'CA[ACGT][ACGT]TG',
    'CBF_DRE': 'GTCGAC',
    
    # Simple 4-mers (common regulatory elements)
    'ACGT': 'ACGT',
    'CACG': 'CACG',
    'CATG': 'CATG',
    'CGTG': 'CGTG',
    'GCGT': 'GCGT',
    'AACA': 'AACA',
    'CAAC': 'CAAC',
    'AAAG': 'AAAG',
    'GATA': 'GATA',
    'TATA': 'TATA',
    'CAAT': 'CAAT',
    'GCCG': 'GCCG',
    'CGCG': 'CGCG',
    'TGAC': 'TGAC',
    'GTAC': 'GTAC',
}


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def reverse_complement(seq: str) -> str:
    """Return reverse complement of a DNA sequence."""
    comp = {"A": "T", "T": "A", "G": "C", "C": "G", "N": "N"}
    return "".join(comp.get(b, "N") for b in reversed(seq.upper()))


def count_motif(sequence: str, pattern: str) -> int:
    """Count motif occurrences on both forward and reverse-complement strands."""
    if pd.isna(sequence) or not isinstance(sequence, str) or len(sequence) == 0:
        return 0
    seq = sequence.upper()
    forward = len(re.findall(pattern, seq))
    reverse = len(re.findall(pattern, reverse_complement(seq)))
    return forward + reverse


def get_seq_stats(sequence: str) -> Tuple[int, float, float, float]:
    """Calculate sequence statistics: length, gc%, at%, gc_skew."""
    if pd.isna(sequence) or not isinstance(sequence, str) or len(sequence) == 0:
        return 0, 0.0, 0.0, 0.0
    seq = sequence.upper()
    length = len(seq)
    g = seq.count("G")
    c = seq.count("C")
    a = seq.count("A")
    t = seq.count("T")

    gc = (g + c) / length * 100.0 if length else 0.0
    at = (a + t) / length * 100.0 if length else 0.0
    gc_skew = (g - c) / (g + c + 1.0)  # +1 for stability

    return length, round(gc, 4), round(at, 4), round(gc_skew, 6)


def scan_sequence_global(sequence: str) -> Dict[str, Any]:
    """
    Scan a promoter sequence for all motifs GLOBALLY.
    Returns total counts for each motif (no positional information).
    """
    counts: Dict[str, Any] = {}

    # Count each motif across the entire sequence
    for motif_name, pattern in MOTIFS.items():
        counts[motif_name] = count_motif(sequence, pattern)

    # Add sequence statistics
    length, gc, at, skew = get_seq_stats(sequence)
    counts["length"] = length
    counts["gc_content"] = gc
    counts["at_content"] = at
    counts["gc_skew"] = skew

    return counts


# =============================================================================
# DATA PROCESSING
# =============================================================================

def load_and_clean_data(filepath: str, top_k: int) -> pd.DataFrame:
    """Load, clean, and filter the rice drought dataset for top-K varieties."""
    print(f"Loading data from: {filepath}")
    df = pd.read_csv(filepath)
    print(f"  Original shape: {df.shape}")

    # Standardize column names
    df.columns = df.columns.str.strip()

    # Convert Gene Regulation -> binary label
    df["label"] = (df["Gene Regulation"] == "Up Regulated").astype(int)

    # Drop missing promoter sequences
    df = df.dropna(subset=["promoter_sequence"])

    # Remove very short sequences (< 100 bp)
    df = df[df["promoter_sequence"].astype(str).str.len() >= 100]

    print(f"  Cleaned shape (before variety filter): {df.shape}")

    # Get top-K varieties by sample count
    variety_counts = df["Variety"].value_counts()
    print("\n  Full Variety distribution:")
    print(variety_counts)
    
    top_k_varieties = variety_counts.head(top_k).index.tolist()
    print(f"\n  Selecting TOP {top_k} varieties: {top_k_varieties}")
    
    # Filter for top-K varieties only
    df = df[df["Variety"].isin(top_k_varieties)]
    print(f"  Filtered shape (top {top_k} varieties): {df.shape}")

    print(f"\n  Selected Variety distribution:")
    print(df["Variety"].value_counts())

    print("\n  Tissue distribution:")
    print(df["Tissue"].value_counts())

    print("\n  Stage distribution:")
    print(df["Stage"].value_counts())

    print("\n  Label distribution (0=Down, 1=Up):")
    print(df["label"].value_counts())

    return df, top_k_varieties


def create_global_motif_matrix(df: pd.DataFrame) -> pd.DataFrame:
    """Create global motif features (no binning)."""
    print(f"\nScanning {len(df)} sequences for {len(MOTIFS)} motifs (global counts)...")

    rows: List[Dict[str, Any]] = []
    total = len(df)
    
    for idx, (_, row) in enumerate(df.iterrows()):
        if (idx + 1) % 5000 == 0:
            print(f"  Processed {idx + 1}/{total} sequences ({(idx+1)/total*100:.1f}%)...")

        counts = scan_sequence_global(row["promoter_sequence"])
        counts["Variety"] = row["Variety"]
        counts["Tissue"] = row["Tissue"]
        counts["Stage"] = row["Stage"]
        counts["label"] = row["label"]
        rows.append(counts)

    result_df = pd.DataFrame(rows)
    print(f"  Done! Matrix shape: {result_df.shape}")
    return result_df


def encode_conditions(df: pd.DataFrame) -> pd.DataFrame:
    """One-hot encode Variety, Tissue, and Stage columns."""
    print("\nEncoding Variety, Tissue, and Stage conditions...")

    variety_dummies = pd.get_dummies(df["Variety"], prefix="Variety")
    tissue_dummies = pd.get_dummies(df["Tissue"], prefix="Tissue")
    stage_dummies = pd.get_dummies(df["Stage"], prefix="Stage")

    # Get motif columns (everything except Variety/Tissue/Stage/label)
    motif_cols = [c for c in df.columns if c not in ["Variety", "Tissue", "Stage", "label"]]

    out = pd.concat(
        [
            df[motif_cols],
            variety_dummies,
            tissue_dummies,
            stage_dummies,
            df[["label"]],
        ],
        axis=1,
    )

    print(f"  Motif + seq stat features: {len(motif_cols)}")
    print(f"  Variety features: {variety_dummies.shape[1]}")
    print(f"  Tissue features: {tissue_dummies.shape[1]}")
    print(f"  Stage features: {stage_dummies.shape[1]}")
    print(f"  Total features: {out.shape[1] - 1}")

    return out


def split_data(df: pd.DataFrame, train_ratio: float = 0.8, dev_ratio: float = 0.1, test_ratio: float = 0.1):
    """Split into train/dev/test (80/10/10) with stratification by label."""
    print(f"\nSplitting data ({train_ratio}:{dev_ratio}:{test_ratio})...")
    
    train_df, temp_df = train_test_split(
        df,
        test_size=(dev_ratio + test_ratio),
        random_state=RANDOM_SEED,
        stratify=df["label"],
    )

    relative_test_ratio = test_ratio / (dev_ratio + test_ratio)
    dev_df, test_df = train_test_split(
        temp_df,
        test_size=relative_test_ratio,
        random_state=RANDOM_SEED,
        stratify=temp_df["label"],
    )

    print(f"  Train: {len(train_df)} samples")
    print(f"  Dev:   {len(dev_df)} samples")
    print(f"  Test:  {len(test_df)} samples")

    for name, split_df in [("Train", train_df), ("Dev", dev_df), ("Test", test_df)]:
        up = int((split_df["label"] == 1).sum())
        down = int((split_df["label"] == 0).sum())
        total = up + down
        pct_up = (up / total * 100.0) if total else 0.0
        print(f"    {name}: Up={up}, Down={down} ({pct_up:.1f}% up)")

    return train_df, dev_df, test_df


# =============================================================================
# MAIN
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description='Rice Motif Pipeline for Top-K Varieties')
    parser.add_argument('--top_k', type=int, required=True, help='Number of top varieties to include (1-13)')
    args = parser.parse_args()
    
    top_k = args.top_k
    
    if top_k < 1 or top_k > 13:
        print(f"ERROR: top_k must be between 1 and 13, got {top_k}")
        sys.exit(1)
    
    OUTPUT_DIR = os.path.join(BASE_OUTPUT_DIR, f"top{top_k}", "data")
    
    print("=" * 80)
    print(f"RICE MOTIF PIPELINE - TOP {top_k} VARIETIES")
    print("=" * 80)

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Step 1: Load, clean, and filter data for top-K varieties
    df, selected_varieties = load_and_clean_data(INPUT_FILE, top_k)
    
    # Step 2: Create global motif matrix (no bins)
    motif_df = create_global_motif_matrix(df)
    
    # Step 3: Encode categorical conditions
    encoded_df = encode_conditions(motif_df)

    # Step 4: Split data
    train_df, dev_df, test_df = split_data(encoded_df)

    # Step 5: Save files
    train_path = os.path.join(OUTPUT_DIR, "train.csv")
    dev_path = os.path.join(OUTPUT_DIR, "dev.csv")
    test_path = os.path.join(OUTPUT_DIR, "test.csv")

    train_df.to_csv(train_path, index=False)
    dev_df.to_csv(dev_path, index=False)
    test_df.to_csv(test_path, index=False)

    # Save feature names
    feature_cols = [c for c in encoded_df.columns if c != "label"]
    with open(os.path.join(OUTPUT_DIR, "feature_names.txt"), "w") as f:
        f.write("\n".join(feature_cols))
    
    # Save selected varieties info
    with open(os.path.join(OUTPUT_DIR, "varieties_info.txt"), "w") as f:
        f.write(f"Top {top_k} Varieties Selected:\n")
        for i, v in enumerate(selected_varieties, 1):
            f.write(f"  {i}. {v}\n")

    print("\nSaved files:")
    print(f"  {train_path}")
    print(f"  {dev_path}")
    print(f"  {test_path}")
    print(f"  {os.path.join(OUTPUT_DIR, 'feature_names.txt')}")
    print(f"  {os.path.join(OUTPUT_DIR, 'varieties_info.txt')}")

    # Feature summary
    motif_cols = list(MOTIFS.keys())
    seq_stat_cols = ['length', 'gc_content', 'at_content', 'gc_skew']
    variety_cols = [c for c in feature_cols if c.startswith("Variety_")]
    tissue_cols = [c for c in feature_cols if c.startswith("Tissue_")]
    stage_cols = [c for c in feature_cols if c.startswith("Stage_")]

    print("\n" + "=" * 80)
    print("FEATURE SUMMARY")
    print("=" * 80)

    print(f"\nMotif features (global counts): {len(motif_cols)}")
    print(f"  Examples: {motif_cols[:10]}")
    
    print(f"\nSequence statistics: {len(seq_stat_cols)}")
    print(f"  {seq_stat_cols}")
    
    print(f"\nVariety one-hot features: {len(variety_cols)}")
    print(f"  {variety_cols}")
    
    print(f"\nTissue one-hot features: {len(tissue_cols)}")
    print(f"  {tissue_cols}")
    
    print(f"\nStage one-hot features: {len(stage_cols)}")
    print(f"  {stage_cols}")

    print("\n" + "=" * 80)
    print(f"TOP {top_k} PIPELINE COMPLETE")
    print("=" * 80)


if __name__ == "__main__":
    main()
