"""
Prepare a patient-level diabetes CSV for baseline training.

Rules:
1) Keep only the first row for each patient_nbr (preserve original file order).
2) Drop encounter_id and patient_nbr columns.
3) Save to a new CSV file (do not overwrite raw data by default).
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


REQUIRED_COLUMNS = ("encounter_id", "patient_nbr")


def parse_args() -> argparse.Namespace:
    base_dir = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input",
        type=Path,
        default=base_dir / "diabetic_data.csv",
        help="Path to raw diabetes CSV.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=base_dir / "diabetic_data_first_encounter.csv",
        help="Path to output CSV.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    df = pd.read_csv(args.input)

    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    n_rows_before = len(df)
    n_patients_before = df["patient_nbr"].nunique()

    # Keep first occurrence in original row order.
    df_first = df.drop_duplicates(subset=["patient_nbr"], keep="first")
    df_out = df_first.drop(columns=["encounter_id", "patient_nbr"])

    args.output.parent.mkdir(parents=True, exist_ok=True)
    df_out.to_csv(args.output, index=False)

    print(f"Input : {args.input}")
    print(f"Output: {args.output}")
    print(f"Rows before: {n_rows_before}")
    print(f"Unique patient_nbr before: {n_patients_before}")
    print(f"Rows after (first per patient): {len(df_first)}")
    print(f"Columns before: {df.shape[1]}")
    print(f"Columns after : {df_out.shape[1]}")

    if "readmitted" in df.columns and "readmitted" in df_out.columns:
        print("\nreadmitted distribution (before):")
        print(df["readmitted"].value_counts(dropna=False).to_dict())
        print("readmitted distribution (after):")
        print(df_out["readmitted"].value_counts(dropna=False).to_dict())


if __name__ == "__main__":
    main()
