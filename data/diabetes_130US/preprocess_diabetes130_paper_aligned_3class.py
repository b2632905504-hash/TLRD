#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Preprocess Diabetes 130-US (UCI/Health Facts) to align with Strack et al. (2014),
BUT keep 3-class label readmitted ∈ {<30, >30, NO}.

Key alignments (paper-ish):
- Keep only valid readmitted labels (<30, >30, NO) [we keep 3-class]
- Remove hospice/death discharges (either via IDs_mapping.csv or a default ID set)
- Keep FIRST encounter per patient_nbr (independence)
- Drop weight (very missing) and payer_code (paper removed)
- Fill medical_specialty missing as "missing" (paper kept it)
- Map diagnosis columns (diag_1/diag_2/diag_3) IN-PLACE into coarse ICD9 categories
  (paper-style bins) and optionally merge groups with <3.5% into "Other"
- Re-bin age IN-PLACE into 3 groups: <30 / 30-60 / >60 (paper used 3 bins)

Notes:
- No extra mapped columns are created (no diag_1_group / age_3bin)
- gender is NOT dropped by default (paper says it wasn't significant; that doesn't strictly mean removed from dataset),
  but you can drop it via --drop_gender

Usage (run in same folder as diabetic_data.csv by default):
  python -u preprocess_diabetes130_paper_aligned_3class.py

If you have IDs_mapping.csv in the same folder and want discharge removal by semantics:
  python -u preprocess_diabetes130_paper_aligned_3class.py --use_mapping_for_discharge

Outputs (default in current working directory):
  ./diabetic_data_paper_aligned_3class.csv
  ./diabetic_data_paper_aligned_3class_report.txt
"""

from __future__ import annotations

import argparse
import io
from pathlib import Path
import pandas as pd

# Common “death/hospice” discharge dispositions used in many reproductions.
# If you have IDs_mapping.csv, prefer --use_mapping_for_discharge.
DEFAULT_BAD_DISPOSITIONS = {11, 13, 14, 19, 20, 21}


def bad_dispositions_from_mapping(mapping_csv: Path) -> set[int]:
    """
    Read IDS_mapping.csv (UCI-style) and derive discharge_disposition_id values that correspond
    to hospice/death/expired by keyword match in the description.

    This version is robust to:
    - file name (IDS vs IDs) handled by CLI/default, not here
    - column naming variations
    - mapping files that contain multiple ID mappings (admission_type_id, admission_source_id, etc.)
      in separate blocks: we try to locate a discharge disposition id column.
    """
    def find_cols(df: pd.DataFrame) -> tuple[str | None, str | None]:
        # Try exact names first
        id_col = "discharge_disposition_id" if "discharge_disposition_id" in df.columns else None
        if id_col is None:
            id_candidates = [c for c in df.columns if ("discharge" in c.lower() and "id" in c.lower())]
            id_col = id_candidates[0] if id_candidates else None

        desc_col = "description" if "description" in df.columns else None
        if desc_col is None:
            desc_candidates = [c for c in df.columns if "desc" in c.lower() or "description" in c.lower()]
            desc_col = desc_candidates[0] if desc_candidates else None
        return id_col, desc_col

    m = pd.read_csv(mapping_csv)
    id_col, desc_col = find_cols(m)

    # UCI IDS_mapping.csv is often a concatenated file with multiple sections separated by blank lines.
    # If the first read does not expose discharge columns, parse section by section.
    if id_col is None:
        with open(mapping_csv, "r", encoding="utf-8") as f:
            raw_lines = f.readlines()

        blocks: list[str] = []
        cur: list[str] = []
        for line in raw_lines:
            stripped = line.strip()
            is_separator = (stripped == "") or (stripped == ",")
            if not is_separator:
                cur.append(line)
            else:
                if cur:
                    blocks.append("".join(cur))
                    cur = []
        if cur:
            blocks.append("".join(cur))

        for block in blocks:
            try:
                bdf = pd.read_csv(io.StringIO(block))
            except Exception:
                continue
            bid, bdesc = find_cols(bdf)
            if bid is not None and bdesc is not None:
                m = bdf
                id_col, desc_col = bid, bdesc
                break

    if id_col is None:
        raise ValueError(
            f"Cannot find a discharge disposition id column in {mapping_csv.name}. "
            f"I see columns={m.columns.tolist()}. "
            f"This looks like a different mapping (e.g., admission_type_id). "
            f"Fix: pass a mapping file that contains discharge_disposition_id, "
            f"or skip --use_mapping_for_discharge and use --bad_dispositions."
        )

    if desc_col is None:
        raise ValueError(
            f"Cannot find a description column in {mapping_csv.name}. Columns={m.columns.tolist()}."
        )

    desc = m[desc_col].astype(str).str.lower()
    keywords = ["hospice", "expired", "death"]

    mask = False
    for kw in keywords:
        mask = mask | desc.str.contains(kw, na=False)

    # Some mapping files have NaNs in rows unrelated to this id type; drop NaNs before casting
    bad_series = m.loc[mask, id_col].dropna()

    # Cast to int safely
    bad = set(bad_series.astype(int).tolist())
    return bad


def map_primary_diag_to_paper_group(code: str) -> str:
    """
    Map ICD-9 code (diag_1) to coarse paper-style grouped diagnosis categories.

    Paper groups emphasize (Table 2 / model footnotes):
      - Circulatory: 390–459, 785
      - Respiratory: 460–519, 786
      - Digestive:   520–579, 787
      - Genitourinary: 580–629, 788
      - Diabetes: 250.*
      - Injury: 800–999
      - Musculoskeletal: 710–739
      - Neoplasms: 140–239
      - Other: everything else (incl. V/E codes, rare groups)
    """
    if code is None:
        return "Other"
    s = str(code).strip()
    if s in {"", "?", "nan", "None"}:
        return "Other"

    su = s.upper()
    # V and E codes → Other (paper lumps many categories into Other)
    if su.startswith("V") or su.startswith("E"):
        return "Other"

    try:
        x = float(s)
    except Exception:
        return "Other"

    # Diabetes special-case
    if 250.0 <= x < 251.0:
        return "Diabetes"

    # Explicit symptom codes used by the paper grouping
    ix = int(x)
    if ix == 785:
        return "Circulatory"
    if ix == 786:
        return "Respiratory"
    if ix == 787:
        return "Digestive"
    if ix == 788:
        return "Genitourinary"

    # Ranges
    if 390.0 <= x < 460.0:
        return "Circulatory"
    if 460.0 <= x < 520.0:
        return "Respiratory"
    if 520.0 <= x < 580.0:
        return "Digestive"
    if 580.0 <= x < 630.0:
        return "Genitourinary"
    if 800.0 <= x < 1000.0:
        return "Injury"
    if 710.0 <= x < 740.0:
        return "Musculoskeletal"
    if 140.0 <= x < 240.0:
        return "Neoplasms"

    return "Other"


def age_to_three_bins(age_str: str) -> str:
    """
    Convert UCI-style age bucket (e.g., '[70-80)') to paper's 3 bins:
      - '<30'
      - '30-60'
      - '>60'
    """
    s = str(age_str).strip()
    try:
        left = int(s.split("[", 1)[1].split("-", 1)[0])
    except Exception:
        return "Unknown"

    if left < 30:
        return "<30"
    if left < 60:
        return "30-60"
    return ">60"


def preprocess(
    in_csv: Path,
    out_csv: Path,
    report_path: Path,
    bad_dispositions: set[int],
    merge_rare_diag_to_other: bool = True,
    rare_thresh: float = 0.035,  # paper says groups <3.5% grouped into Other
    drop_gender: bool = False,
) -> None:
    df = pd.read_csv(in_csv)

    lines: list[str] = []

    def log(msg: str) -> None:
        print(msg)
        lines.append(msg)

    log(f"[Load] {in_csv.resolve()}")
    log(f"  rows={len(df):,}, cols={df.shape[1]}")
    log(f"[CWD] {Path.cwd().resolve()}")

    # --- 0) Label normalization (keep 3-class) ---
    if "readmitted" not in df.columns:
        raise ValueError("Column 'readmitted' not found in input CSV.")

    df["readmitted"] = df["readmitted"].astype(str).str.strip().str.upper()
    before = len(df)
    df = df[df["readmitted"].isin(["<30", ">30", "NO"])].copy()
    log(f"[Label] Keep 3-class (<30, >30, NO): {before:,} -> {len(df):,}")
    log("  label counts:\n" + df["readmitted"].value_counts().to_string())

    # --- 1) Remove hospice/death discharge dispositions ---
    if "discharge_disposition_id" in df.columns:
        log(f"[Discharge] bad_dispositions used: {sorted(bad_dispositions)}")
        before = len(df)
        df = df[~df["discharge_disposition_id"].isin(bad_dispositions)].copy()
        log(f"[Discharge] Removed: {before:,} -> {len(df):,}")
    else:
        log("[Discharge] WARNING: 'discharge_disposition_id' not found; skipping this step.")

    # --- 2) Keep first encounter per patient (independence) ---
    needed = {"patient_nbr", "encounter_id"}
    if needed.issubset(df.columns):
        before = len(df)
        df = df.sort_values(["patient_nbr", "encounter_id"])
        df = df.drop_duplicates(subset=["patient_nbr"], keep="first").copy()
        log(f"[Independence] Keep first encounter per patient: {before:,} -> {len(df):,}")
    else:
        log(f"[Independence] WARNING: missing {sorted(list(needed - set(df.columns)))}; skipping step.")

    # --- 3) Drop columns per paper ---
    for col in ["weight", "payer_code"]:
        if col in df.columns:
            df = df.drop(columns=[col])
            log(f"[Drop] Dropped column: {col}")
        else:
            log(f"[Drop] Column not found (ok): {col}")

    if drop_gender and "gender" in df.columns:
        df = df.drop(columns=["gender"])
        log("[Drop] Dropped column: gender")

    # --- 4) Fill medical_specialty missing as 'missing' ---
    if "medical_specialty" in df.columns:
        before_missing = (
            df["medical_specialty"].isna()
            | (df["medical_specialty"].astype(str).str.strip() == "?")
        ).sum()
        df["medical_specialty"] = df["medical_specialty"].fillna("missing")
        df["medical_specialty"] = df["medical_specialty"].replace("?", "missing")
        after_missing = (df["medical_specialty"].astype(str).str.strip() == "missing").sum()
        log(f"[Missing] medical_specialty: filled {before_missing:,} -> now 'missing' count={after_missing:,}")
    else:
        log("[Missing] WARNING: 'medical_specialty' not found; skipping fill.")

    # --- 5) Diagnosis grouping on diag_1/diag_2/diag_3 (in-place) ---
    diag_cols = [c for c in ["diag_1", "diag_2", "diag_3"] if c in df.columns]
    if diag_cols:
        for col in diag_cols:
            df[col] = df[col].apply(map_primary_diag_to_paper_group)
        log(f"[Diag] Mapped diagnosis columns in-place: {diag_cols}")

        # Optional: merge rare diagnosis groups (<3.5%) into Other
        if merge_rare_diag_to_other:
            for col in diag_cols:
                freq = df[col].value_counts(normalize=True)
                rare_groups = set(freq[freq < rare_thresh].index)
                rare_groups.discard("Other")
                if rare_groups:
                    df.loc[df[col].isin(rare_groups), col] = "Other"
                    log(
                        f"[Diag] Merged rare mapped {col} groups (<{rare_thresh*100:.1f}%) into 'Other': "
                        f"{sorted(list(rare_groups))}"
                    )
                else:
                    log(f"[Diag] No mapped {col} group under {rare_thresh*100:.1f}% to merge.")
        else:
            log("[Diag] Mapping done (no rare-merge).")

        for col in diag_cols:
            log(f"[Diag] Mapped {col} counts:\n" + df[col].value_counts().to_string())
    else:
        log("[Diag] WARNING: none of ['diag_1', 'diag_2', 'diag_3'] found; skipping diagnosis grouping.")

    # --- 6) Age regrouping to 3 bins (in-place) ---
    if "age" in df.columns:
        df["age"] = df["age"].apply(age_to_three_bins)
        log("[Age] Mapped age in-place to {<30, 30-60, >60, Unknown}")
        log("[Age] age counts (mapped):\n" + df["age"].value_counts().to_string())
    else:
        log("[Age] WARNING: 'age' not found; skipping age regrouping.")

    # --- 7) Drop ID columns before export (avoid identifier leakage in modeling) ---
    id_cols = [c for c in ["encounter_id", "patient_nbr"] if c in df.columns]
    if id_cols:
        df = df.drop(columns=id_cols)
        log(f"[Drop] Dropped ID columns: {id_cols}")
    else:
        log("[Drop] ID columns not found (ok): ['encounter_id', 'patient_nbr']")

    # --- Save outputs ---
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_csv, index=False)
    log(f"[Save] Wrote: {out_csv.resolve()}  (rows={len(df):,}, cols={df.shape[1]})")

    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    log(f"[Report] Wrote: {report_path.resolve()}")


def main() -> None:
    ap = argparse.ArgumentParser()

    # Default to current working directory files (run in same folder as CSV)
    ap.add_argument("--in_csv", type=Path, default=Path("./diabetic_data.csv"))
    ap.add_argument("--out_csv", type=Path, default=Path("./diabetic_data_paper_aligned_3class.csv"))
    ap.add_argument("--report", type=Path, default=Path("./diabetic_data_paper_aligned_3class_report.txt"))

    # Discharge removal: either fixed IDs or derived from IDs_mapping.csv
    ap.add_argument(
        "--bad_dispositions",
        type=str,
        default="11,13,14,19,20,21",
        help="Comma-separated discharge_disposition_id values to REMOVE (death/hospice-like). Ignored if --use_mapping_for_discharge is set.",
    )
    ap.add_argument(
        "--ids_mapping",
        type=Path,
        default=Path("./IDS_mapping.csv"),
        help="Path to IDs_mapping.csv (same folder as diabetic_data.csv by default).",
    )
    ap.add_argument(
        "--use_mapping_for_discharge",
        action="store_true",
        help="Derive bad discharge_disposition_id from IDs_mapping.csv using hospice/death/expired keywords.",
    )

    # Diagnosis grouping controls
    ap.add_argument(
        "--no_merge_rare_diag",
        action="store_true",
        help="Disable merging rare mapped diagnosis groups (<3.5%) into Other.",
    )
    ap.add_argument(
        "--rare_thresh",
        type=float,
        default=0.035,
        help="Threshold for merging rare mapped diagnosis groups into Other (default 0.035 = 3.5%).",
    )
    ap.add_argument(
        "--keep_raw_diag",
        action="store_true",
        help="Deprecated no-op. Mapping is now in-place on diag_1 and diag_2/diag_3 are kept.",
    )

    # Age controls
    ap.add_argument(
        "--keep_raw_age",
        action="store_true",
        help="Deprecated no-op. Mapping is now in-place on age.",
    )

    # Optional drop
    ap.add_argument(
        "--drop_gender",
        action="store_true",
        help="Drop gender column (optional).",
    )

    args = ap.parse_args()

    if args.keep_raw_diag:
        print("[Args] NOTE: --keep_raw_diag is deprecated and ignored.")
    if args.keep_raw_age:
        print("[Args] NOTE: --keep_raw_age is deprecated and ignored.")

    # Determine bad disposition IDs
    if args.use_mapping_for_discharge:
        if not args.ids_mapping.exists():
            raise FileNotFoundError(
                f"--use_mapping_for_discharge was set, but IDs_mapping.csv not found at: {args.ids_mapping}"
            )
        bad = bad_dispositions_from_mapping(args.ids_mapping)
    else:
        bad = (
            {int(x.strip()) for x in args.bad_dispositions.split(",") if x.strip()}
            if args.bad_dispositions.strip()
            else set(DEFAULT_BAD_DISPOSITIONS)
        )

    preprocess(
        in_csv=args.in_csv,
        out_csv=args.out_csv,
        report_path=args.report,
        bad_dispositions=bad if bad else set(DEFAULT_BAD_DISPOSITIONS),
        merge_rare_diag_to_other=not args.no_merge_rare_diag,
        rare_thresh=args.rare_thresh,
        drop_gender=args.drop_gender,
    )


if __name__ == "__main__":
    main()
