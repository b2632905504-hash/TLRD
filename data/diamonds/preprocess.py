"""
Diamonds Dataset Preprocessing
"""

import argparse
import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import train_test_split

# Dataset config
CSV_PATH = "diamonds.csv"
TARGET_COL = "price"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    # Load data
    df = pd.read_csv(CSV_PATH)
    print(f"Loaded {len(df)} samples from {CSV_PATH}")

    # Print basic statistics about the target
    print(f"\nTarget column: {TARGET_COL}")
    print(f"  Min: ${df[TARGET_COL].min():,.0f}")
    print(f"  Max: ${df[TARGET_COL].max():,.0f}")
    print(f"  Mean: ${df[TARGET_COL].mean():,.0f}")
    print(f"  Median: ${df[TARGET_COL].median():,.0f}")

    # For regression task, we use stratified splitting based on price quartiles
    # to ensure similar price distributions across splits
    y_reg = df[TARGET_COL]
    y_bins = pd.qcut(y_reg, q=10, labels=False, duplicates='drop')  # 10 bins for stratification
    idx_all = np.arange(len(df))

    # Split 8:1:1 with stratification on price bins
    idx_temp, idx_test = train_test_split(
        idx_all, test_size=0.1, random_state=args.seed, stratify=y_bins
    )
    idx_train, idx_valid = train_test_split(
        idx_temp, test_size=1/9, random_state=args.seed, stratify=y_bins[idx_temp]
    )

    # Print stats
    print(f"\nSplit (seed={args.seed}):")
    print(f"  Train: {len(idx_train)} ({100*len(idx_train)/len(df):.1f}%)")
    print(f"  Valid: {len(idx_valid)} ({100*len(idx_valid)/len(df):.1f}%)")
    print(f"  Test:  {len(idx_test)} ({100*len(idx_test)/len(df):.1f}%)")

    # Price distribution per split
    for name, idx in [("Train", idx_train), ("Valid", idx_valid), ("Test", idx_test)]:
        y_split = df[TARGET_COL].iloc[idx]
        print(f"\n{name} price distribution:")
        print(f"  Min: ${y_split.min():,.0f}")
        print(f"  Max: ${y_split.max():,.0f}")
        print(f"  Mean: ${y_split.mean():,.0f}")
        print(f"  Median: ${y_split.median():,.0f}")

    # Save
    output_path = f"split_{args.seed}.pth"
    torch.save({
        "idx_train": torch.tensor(idx_train, dtype=torch.long),
        "idx_valid": torch.tensor(idx_valid, dtype=torch.long),
        "idx_test": torch.tensor(idx_test, dtype=torch.long),
        "seed": args.seed,
    }, output_path)
    print(f"\nSaved to: {output_path}")


if __name__ == "__main__":
    main()

