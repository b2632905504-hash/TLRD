"""
Diabetes 130-US preprocessing
Generates train/valid/test splits (8:1:1) and saves to split_{seed}.pth
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import train_test_split


CSV_PATH = "diabetic_data_paper_aligned_3class.csv"
TARGET_COL = "readmitted"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    data_path = Path(__file__).resolve().parent / CSV_PATH
    df = pd.read_csv(data_path)
    print(f"Loaded {len(df)} samples from {data_path}")

    y = df[TARGET_COL]
    idx_all = np.arange(len(df))

    idx_temp, idx_test = train_test_split(
        idx_all,
        test_size=0.1,
        random_state=args.seed,
        stratify=y,
    )
    idx_train, idx_valid = train_test_split(
        idx_temp,
        test_size=1 / 9,
        random_state=args.seed,
        stratify=y.iloc[idx_temp],
    )

    print(f"\nSplit (seed={args.seed}):")
    print(f"  Train: {len(idx_train)} ({100 * len(idx_train) / len(df):.1f}%)")
    print(f"  Valid: {len(idx_valid)} ({100 * len(idx_valid) / len(df):.1f}%)")
    print(f"  Test:  {len(idx_test)} ({100 * len(idx_test) / len(df):.1f}%)")

    for name, idx in [("Train", idx_train), ("Valid", idx_valid), ("Test", idx_test)]:
        y_split = y.iloc[idx]
        counts = y_split.value_counts()
        print(f"\n{name} class distribution:")
        for val, cnt in counts.items():
            print(f"  {val}: {cnt} ({100 * cnt / len(idx):.2f}%)")

    output_path = data_path.parent / f"split_{args.seed}.pth"
    torch.save(
        {
            "idx_train": torch.tensor(idx_train, dtype=torch.long),
            "idx_valid": torch.tensor(idx_valid, dtype=torch.long),
            "idx_test": torch.tensor(idx_test, dtype=torch.long),
            "seed": args.seed,
        },
        output_path,
    )
    print(f"\nSaved to: {output_path}")


if __name__ == "__main__":
    main()
