"""
Dataset statistics utilities used by corpus_builder.

Computes overall or per-class statistics from the training split and formats
them for LLM prompts. Separated into its own module to keep corpus_builder
focused on orchestration.
"""

import json
from typing import List, Dict, Any, Optional
from pathlib import Path
import sys

import numpy as np
import pandas as pd

from data.dataset import TabularDataset

# Ensure local imports work when executed as a script
_SCRIPT_DIR = Path(__file__).resolve().parent
if str(_SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPT_DIR))
_PARENT_DIR = _SCRIPT_DIR.parent
if str(_PARENT_DIR) not in sys.path:
    sys.path.insert(0, str(_PARENT_DIR))


# Ablation variants for numerical feature statistics
# Each variant specifies which stats to include in the output
ABLATION_VARIANTS = {
    "mean_only": ["mean"],                              # only mean
    "mean_std": ["mean", "std"],                        # mean + std
    "no_minmax": ["mean", "std", "p25", "p50", "p75"],  # exclude min/max
    "full": ["mean", "std", "p25", "p50", "p75", "min", "max"],  # full set
}


class DatasetStats:
    """
    Compute and store statistics from training data only.

    This ensures no data leakage when building corpus for any split.
    Statistics include:
    - Target class distribution
    - Numerical features: mean, std, min, max, quartiles
    - Categorical features: top-k categories with percentages

    Optionally computes per-class statistics for better interpretability.
    """

    def __init__(
        self,
        train_dataset: TabularDataset,
        top_k: Optional[int] = 5,
        ablation_variant: str = "full",
        per_class: bool = False,
    ):
        """
        Args:
            train_dataset: TabularDataset with split="train"
            top_k: Number of top categories to include for categorical features
                (None means include all categories, default: 5)
            ablation_variant: Which stats to include for numerical features.
                Options: "mean_only", "mean_std", "no_minmax", "full"
            per_class: Whether to compute statistics separately for each target class
        """
        if ablation_variant not in ABLATION_VARIANTS:
            raise ValueError(
                f"Invalid ablation_variant '{ablation_variant}'. "
                f"Available: {list(ABLATION_VARIANTS.keys())}"
            )

        self.train_df = train_dataset.df.copy()
        self.num_cols = train_dataset.num_cols
        self.cat_cols = train_dataset.cat_cols
        self.target_col = train_dataset.target_col
        target_display_name = getattr(train_dataset.config, "target_display_name", None)
        self.target_display_name = (
            str(target_display_name) if target_display_name else self.target_col
        )
        self.top_k = top_k
        self.ablation_variant = ablation_variant
        self.per_class = per_class
        self.n_train = len(train_dataset)
        self.task_type = train_dataset.task_type  # Determine if regression
        self.dataset_name = train_dataset.config.name  # For special cases
        self.target_label_display_map = {
            str(k): str(v)
            for k, v in (train_dataset.config.target_label_display_map or {}).items()
        }

        # Get unique target classes
        self.target_classes = list(self.train_df[self.target_col].unique())

        self._compute_stats()
        if self.per_class:
            if self.task_type == "regression":
                self._compute_per_bin_stats()
            else:
                self._compute_per_class_stats()

    def display_target_label(self, label: str) -> str:
        """Convert raw target label to display label if mapping exists."""
        return self.target_label_display_map.get(str(label), str(label))

    def _compute_stats(self):
        """Compute all statistics from training data."""
        # Numerical feature statistics
        self.num_stats: Dict[str, Dict[str, float]] = {}
        for col in self.num_cols:
            s = self.train_df[col].dropna()
            if len(s) == 0:
                continue
            self.num_stats[col] = {
                "mean": float(s.mean()),
                "std": float(s.std()),
                "min": float(s.min()),
                "max": float(s.max()),
                "p25": float(s.quantile(0.25)),
                "p50": float(s.quantile(0.50)),
                "p75": float(s.quantile(0.75)),
                "n_missing": int(self.train_df[col].isna().sum()),
            }

        # Categorical feature statistics
        self.cat_stats: Dict[str, Dict[str, Any]] = {}
        self._cat_value_counts: Dict[str, Dict[str, int]] = {}
        for col in self.cat_cols:
            s = self.train_df[col].fillna("missing")
            s = s.replace("?", "missing")
            vc = s.value_counts()
            self._cat_value_counts[col] = {
                str(k): int(v) for k, v in vc.items()
            }
            top_k_values = vc if self.top_k is None else vc.head(self.top_k)
            total = len(s)
            self.cat_stats[col] = {
                "top_values": list(top_k_values.index),
                "top_counts": [int(x) for x in top_k_values.values],
                "top_proportions": [
                    float(x / total) if total > 0 else 0.0
                    for x in top_k_values.values
                ],
                "n_unique": int(s.nunique()),
                "n_missing": int(self.train_df[col].isna().sum() + (self.train_df[col] == "?").sum()),
            }

        # Target distribution (only for classification tasks)
        # Special case: California dataset with classification task (target converted to above_median/below_median)

        if self.task_type != "regression":
            target_vc = self.train_df[self.target_col].value_counts()
            self.target_dist = {str(k): int(v) for k, v in target_vc.items()}
            self.target_proportions = {str(k): float(v / len(self.train_df)) for k, v in target_vc.items()}
        else:
            self.target_dist = {}
            self.target_proportions = {}

    def _compute_per_class_stats(self):
        """Compute statistics separately for each target class."""
        # Per-class numerical statistics: {class_label: {col: {stat: value}}}
        self.per_class_num_stats: Dict[str, Dict[str, Dict[str, float]]] = {}
        # Per-class categorical statistics: {class_label: {col: {stat: value}}}
        self.per_class_cat_stats: Dict[str, Dict[str, Dict[str, Any]]] = {}
        self._per_class_cat_value_counts: Dict[str, Dict[str, Dict[str, int]]] = {}
        self._per_class_sample_counts: Dict[str, int] = {}
        self._per_class_sample_proportions: Dict[str, float] = {}

        for target_class in self.target_classes:
            class_label = str(target_class)
            class_df = self.train_df[self.train_df[self.target_col] == target_class]
            self._per_class_sample_counts[class_label] = int(len(class_df))
            self._per_class_sample_proportions[class_label] = (
                float(len(class_df) / self.n_train) if self.n_train > 0 else 0.0
            )

            # Numerical features for this class
            self.per_class_num_stats[class_label] = {}
            for col in self.num_cols:
                s = class_df[col].dropna()
                if len(s) == 0:
                    continue
                self.per_class_num_stats[class_label][col] = {
                    "mean": float(s.mean()),
                    "std": float(s.std()),
                    "min": float(s.min()),
                    "max": float(s.max()),
                    "p25": float(s.quantile(0.25)),
                    "p50": float(s.quantile(0.50)),
                    "p75": float(s.quantile(0.75)),
                    "n_samples": len(s),
                }

            # Categorical features for this class
            self.per_class_cat_stats[class_label] = {}
            self._per_class_cat_value_counts[class_label] = {}
            for col in self.cat_cols:
                s = class_df[col].fillna("missing").replace("?", "missing")
                vc = s.value_counts()
                self._per_class_cat_value_counts[class_label][col] = {
                    str(k): int(v) for k, v in vc.items()
                }
                top_k_values = vc if self.top_k is None else vc.head(self.top_k)
                total = len(s)
                self.per_class_cat_stats[class_label][col] = {
                    "top_values": list(top_k_values.index),
                    "top_counts": [int(x) for x in top_k_values.values],
                    "top_proportions": [
                        float(x / total) if total > 0 else 0.0
                        for x in top_k_values.values
                    ],
                    "n_unique": int(s.nunique()),
                }

    def _compute_per_bin_stats(self):
        """Compute statistics for each target percentile bin (regression tasks).

        Bins the target into 4 quartile-based bins using the 25th, 50th and
        75th percentiles and computes per-bin numerical and categorical stats,
        mirroring the structure of ``_compute_per_class_stats``.
        """
        target_series = self.train_df[self.target_col].dropna()

        # Compute quartile edges
        percentiles = [0, 25, 50, 75, 100]
        bin_edges = list(np.percentile(target_series, percentiles))

        # Remove duplicate edges (can happen when many values are identical)
        bin_edges_unique = sorted(set(bin_edges))

        assert len(bin_edges_unique) >=4 

        # Bin the target column using pd.cut
        self.train_df["_target_bin"] = pd.cut(
            self.train_df[self.target_col],
            bins=bin_edges_unique,
            include_lowest=True,
            duplicates="drop",
        )

        bin_intervals = sorted(self.train_df["_target_bin"].dropna().unique())
        self.bin_labels = [str(b) for b in bin_intervals]
        self.bin_edges = bin_edges_unique

        # Store the raw percentile values for display
        self.bin_percentiles = {
            "p0": float(np.percentile(target_series, 0)),
            "p25": float(np.percentile(target_series, 25)),
            "p50": float(np.percentile(target_series, 50)),
            "p75": float(np.percentile(target_series, 75)),
            "p100": float(np.percentile(target_series, 100)),
        }

        # Build display ranges for each bin using original bin edges.
        # This avoids pd.cut's include_lowest display artifact like "14998.999".
        edge_to_percentiles: Dict[float, List[float]] = {}
        for edge in bin_edges_unique:
            matched = [
                float(pct) for pct, raw_edge in zip(percentiles, bin_edges)
                if np.isclose(raw_edge, edge)
            ]
            if matched:
                edge_to_percentiles[float(edge)] = matched

        self.bin_value_ranges = []
        self.bin_percentile_ranges = []
        for i in range(len(bin_edges_unique) - 1):
            left = float(bin_edges_unique[i])
            right = float(bin_edges_unique[i + 1])
            left_pcts = edge_to_percentiles.get(left, [0.0])
            right_pcts = edge_to_percentiles.get(right, [100.0])
            # Use the max percentile on each boundary so merged bins remain contiguous.
            pct_low = float(max(left_pcts))
            pct_high = float(max(right_pcts))
            self.bin_value_ranges.append((left, right))
            self.bin_percentile_ranges.append((pct_low, pct_high))

        # Per-bin statistics (same structure as per-class)
        self.per_bin_num_stats = {}
        self.per_bin_cat_stats = {}
        self._per_bin_cat_value_counts = {}
        self._per_bin_sample_counts = {}
        self._per_bin_sample_proportions = {}

        for interval, bin_label in zip(bin_intervals, self.bin_labels):
            bin_df = self.train_df[self.train_df["_target_bin"] == interval]
            self._per_bin_sample_counts[bin_label] = int(len(bin_df))
            self._per_bin_sample_proportions[bin_label] = (
                float(len(bin_df) / self.n_train) if self.n_train > 0 else 0.0
            )

            # Numerical features for this bin
            self.per_bin_num_stats[bin_label] = {}
            for col in self.num_cols:
                s = bin_df[col].dropna()
                if len(s) == 0:
                    continue
                self.per_bin_num_stats[bin_label][col] = {
                    "mean": float(s.mean()),
                    "std": float(s.std()),
                    "min": float(s.min()),
                    "max": float(s.max()),
                    "p25": float(s.quantile(0.25)),
                    "p50": float(s.quantile(0.50)),
                    "p75": float(s.quantile(0.75)),
                    "n_samples": len(s),
                }

            # Categorical features for this bin
            self.per_bin_cat_stats[bin_label] = {}
            self._per_bin_cat_value_counts[bin_label] = {}
            for col in self.cat_cols:
                s = bin_df[col].fillna("missing").replace("?", "missing")
                vc = s.value_counts()
                self._per_bin_cat_value_counts[bin_label][col] = {
                    str(k): int(v) for k, v in vc.items()
                }
                top_k_values = vc if self.top_k is None else vc.head(self.top_k)
                total = len(s)
                self.per_bin_cat_stats[bin_label][col] = {
                    "top_values": list(top_k_values.index),
                    "top_counts": [int(x) for x in top_k_values.values],
                    "top_proportions": [
                        float(x / total) if total > 0 else 0.0
                        for x in top_k_values.values
                    ],
                    "n_unique": int(s.nunique()),
                }

        # Clean up temporary column
        self.train_df.drop(columns=["_target_bin"], inplace=True)

    def _format_num_stats_line(self, s: Dict[str, float], included_stats: List[str]) -> List[str]:
        """Helper to format numerical stats for a single feature."""
        lines = []

        # Build stats line based on ablation variant
        stat_parts = []
        if "mean" in included_stats:
            stat_parts.append(f"mean={s['mean']:.2f}")
        if "std" in included_stats:
            stat_parts.append(f"std={s['std']:.2f}")
        if stat_parts:
            lines.append(f"      {', '.join(stat_parts)}")

        # Percentiles line
        percentile_parts = []
        if "p25" in included_stats:
            percentile_parts.append(f"p25={s['p25']:.2f}")
        if "p50" in included_stats:
            percentile_parts.append(f"p50={s['p50']:.2f}")
        if "p75" in included_stats:
            percentile_parts.append(f"p75={s['p75']:.2f}")
        if percentile_parts:
            lines.append(f"      {', '.join(percentile_parts)}")

        # Min/max line
        minmax_parts = []
        if "min" in included_stats:
            minmax_parts.append(f"min={s['min']:.2f}")
        if "max" in included_stats:
            minmax_parts.append(f"max={s['max']:.2f}")
        if minmax_parts:
            lines.append(f"      {', '.join(minmax_parts)}")

        return lines

    @staticmethod
    def _format_percent_label(value: float) -> str:
        """Format percentile value for display."""
        if np.isclose(value, round(value)):
            return f"{int(round(value))}%"
        return f"{value:.1f}%"

    def format_stats_text(
        self,
        include_header: bool = True,
        current_sample_features: Optional[Dict[str, Any]] = None,
    ) -> str:
        """
        Format statistics as natural language text for LLM prompts.

        Args:
            include_header: Whether to include section header
            current_sample_features: Optional current sample features. If provided,
                categorical sections include the current sample category per feature.

        Returns:
            Formatted statistics text
        """
        lines = []
        included_stats = ABLATION_VARIANTS[self.ablation_variant]

        if include_header:
            if self.per_class and self.task_type == "regression":
                stats_mode = "per-bin"
            elif self.per_class:
                stats_mode = "per-class"
            else:
                stats_mode = "overall"
            lines.append(f"=== Dataset Statistics (from training set, {stats_mode}) ===")
            lines.append(f"Total training samples: {self.n_train}")

        # Target distribution (for classification tasks, including converted regression targets)
        if include_header and self.target_dist:
            lines.append(f"\nTarget distribution ({self.target_display_name}):")
            for label, count in self.target_dist.items():
                prop = self.target_proportions[label]
                label_display = self.display_target_label(label)
                lines.append(f"  {label_display}: {count} ({prop:.1%})")

        if self.per_class and self.task_type == "regression":
            # Per-bin statistics for regression
            p = getattr(self, "bin_percentiles", {})
            p0_str = f"{p.get('p0', 0):.2f}" if p else "?"
            p25_str = f"{p.get('p25', 0):.2f}" if p else "?"
            p50_str = f"{p.get('p50', 0):.2f}" if p else "?"
            p75_str = f"{p.get('p75', 0):.2f}" if p else "?"
            p100_str = f"{p.get('p100', 0):.2f}" if p else "?"
            lines.append(
                f"\n--- Statistics by {self.target_display_name} Percentile Bin "
                f"(p0={p0_str}, p25={p25_str}, p50={p50_str}, p75={p75_str}, p100={p100_str}) ---"
            )

            for bin_idx, bin_label in enumerate(self.bin_labels):
                bin_count = self._per_bin_sample_counts.get(bin_label, 0)
                bin_prop = self._per_bin_sample_proportions.get(bin_label, 0.0)
                q_label = f"Q{bin_idx + 1}"
                value_ranges = getattr(self, "bin_value_ranges", [])
                percentile_ranges = getattr(self, "bin_percentile_ranges", [])
                if bin_idx < len(value_ranges):
                    left, right = value_ranges[bin_idx]
                    range_text = f"({left:.1f}, {right:.1f}]"
                else:
                    range_text = bin_label
                if bin_idx < len(percentile_ranges):
                    pct_low, pct_high = percentile_ranges[bin_idx]
                    pct_text = f"{self._format_percent_label(pct_low)}-{self._format_percent_label(pct_high)}"
                else:
                    pct_text = "?"
                lines.append(
                    f"\n{q_label} ({pct_text}): {self.target_display_name} in {range_text} "
                    f"(n={bin_count}, {bin_prop:.1%})"
                )

                # Numerical features for this bin
                if bin_label in self.per_bin_num_stats and self.per_bin_num_stats[bin_label]:
                    lines.append("  Numerical features:")
                    for col in self.num_cols:
                        if col not in self.per_bin_num_stats[bin_label]:
                            continue
                        s = self.per_bin_num_stats[bin_label][col]
                        lines.append(f"    {col}:")
                        lines.extend(self._format_num_stats_line(s, included_stats))

                # Categorical features for this bin
                if bin_label in self.per_bin_cat_stats and self.per_bin_cat_stats[bin_label]:
                    cat_scope = "all" if self.top_k is None else f"top {self.top_k}"
                    lines.append(f"  Categorical features ({cat_scope}):")
                    for col in self.cat_cols:
                        if col not in self.per_bin_cat_stats[bin_label]:
                            continue
                        s = self.per_bin_cat_stats[bin_label][col]
                        total = self._per_bin_sample_counts.get(bin_label, bin_count)
                        total = total if total > 0 else 1
                        top_str = ", ".join(
                            f"{v}({c / total:.1%})" for v, c in zip(s["top_values"], s["top_counts"])
                        )
                        lines.append(f"    {col}: {top_str}")
                        if current_sample_features is not None and col in current_sample_features:
                            current_sample_value = str(current_sample_features[col])
                            current_sample_count = self._per_bin_cat_value_counts.get(
                                bin_label, {}
                            ).get(col, {}).get(current_sample_value, 0)
                            current_sample_ratio = current_sample_count / total
                            lines.append(
                                f"    current_sample={current_sample_value}({current_sample_ratio:.1%})"
                            )

        elif self.per_class:
            # Per-class statistics (classification)
            lines.append("\n--- Statistics by Target Class ---")

            for target_class in self.target_classes:
                class_label = str(target_class)
                class_label_display = self.display_target_label(class_label)
                class_count = self.target_dist.get(class_label, 0)
                class_prop = self.target_proportions.get(class_label, 0.0)
                lines.append(
                    f"\n[{self.target_display_name} = {class_label_display}] "
                    f"(n={class_count}, {class_prop:.1%}):"
                )

                # Numerical features for this class
                if class_label in self.per_class_num_stats and self.per_class_num_stats[class_label]:
                    lines.append("  Numerical features:")
                    for col in self.num_cols:
                        if col not in self.per_class_num_stats[class_label]:
                            continue
                        s = self.per_class_num_stats[class_label][col]
                        lines.append(f"    {col}:")
                        lines.extend(self._format_num_stats_line(s, included_stats))

                # Categorical features for this class
                if class_label in self.per_class_cat_stats and self.per_class_cat_stats[class_label]:
                    cat_scope = "all" if self.top_k is None else f"top {self.top_k}"
                    lines.append(f"  Categorical features ({cat_scope}):")
                    for col in self.cat_cols:
                        if col not in self.per_class_cat_stats[class_label]:
                            continue
                        s = self.per_class_cat_stats[class_label][col]
                        total = self._per_class_sample_counts.get(class_label, class_count)
                        total = total if total > 0 else 1
                        top_str = ", ".join(
                            f"{v}({c / total:.1%})" for v, c in zip(s["top_values"], s["top_counts"])
                        )
                        lines.append(f"    {col}: {top_str}")
                        if current_sample_features is not None and col in current_sample_features:
                            current_sample_value = str(current_sample_features[col])
                            current_sample_count = self._per_class_cat_value_counts.get(
                                class_label, {}
                            ).get(col, {}).get(current_sample_value, 0)
                            current_sample_ratio = current_sample_count / total
                            lines.append(
                                f"    current_sample={current_sample_value}({current_sample_ratio:.1%})"
                            )
        else:
            # Overall statistics (original behavior)
            # Numerical features
            if self.num_stats:
                lines.append("\nNumerical features:")
                for col in self.num_cols:
                    if col not in self.num_stats:
                        continue
                    s = self.num_stats[col]
                    lines.append(f"  {col}:")

                    # Build stats line based on ablation variant
                    stat_parts = []
                    if "mean" in included_stats:
                        stat_parts.append(f"mean={s['mean']:.2f}")
                    if "std" in included_stats:
                        stat_parts.append(f"std={s['std']:.2f}")
                    if stat_parts:
                        lines.append(f"    {', '.join(stat_parts)}")

                    # Percentiles line
                    percentile_parts = []
                    if "p25" in included_stats:
                        percentile_parts.append(f"p25={s['p25']:.2f}")
                    if "p50" in included_stats:
                        percentile_parts.append(f"p50={s['p50']:.2f}")
                    if "p75" in included_stats:
                        percentile_parts.append(f"p75={s['p75']:.2f}")
                    if percentile_parts:
                        lines.append(f"    {', '.join(percentile_parts)}")

                    # Min/max line
                    minmax_parts = []
                    if "min" in included_stats:
                        minmax_parts.append(f"min={s['min']:.2f}")
                    if "max" in included_stats:
                        minmax_parts.append(f"max={s['max']:.2f}")
                    if minmax_parts:
                        lines.append(f"    {', '.join(minmax_parts)}")

            # Categorical features
            if self.cat_stats:
                cat_scope = "all" if self.top_k is None else f"top {self.top_k}"
                lines.append(f"\nCategorical features ({cat_scope} values):")
                for col in self.cat_cols:
                    if col not in self.cat_stats:
                        continue
                    s = self.cat_stats[col]
                    lines.append(f"  {col} ({s['n_unique']} unique):")
                    total = self.n_train if self.n_train > 0 else 1
                    top_str = ", ".join(
                        f"{v}({c / total:.1%})" for v, c in zip(s["top_values"], s["top_counts"])
                    )
                    lines.append(f"    {top_str}")
                    if current_sample_features is not None and col in current_sample_features:
                        current_sample_value = str(current_sample_features[col])
                        current_sample_count = self._cat_value_counts.get(col, {}).get(current_sample_value, 0)
                        current_sample_ratio = current_sample_count / total
                        lines.append(f"    current_sample={current_sample_value}({current_sample_ratio:.1%})")

        return "\n".join(lines)

    def to_dict(self) -> Dict[str, Any]:
        """Export statistics as a dictionary."""
        result = {
            "n_train": self.n_train,
            "target_col": self.target_col,
            "target_display_name": self.target_display_name,
            "target_classes": [str(c) for c in self.target_classes],
            "target_dist": self.target_dist,
            "target_proportions": self.target_proportions,
            "num_stats": self.num_stats,
            "cat_stats": self.cat_stats,
            "ablation_variant": self.ablation_variant,
            "included_stats": ABLATION_VARIANTS[self.ablation_variant],
            "per_class": self.per_class,
        }

        if self.per_class:
            if self.task_type == "regression":
                result["per_bin_num_stats"] = self.per_bin_num_stats
                result["per_bin_cat_stats"] = self.per_bin_cat_stats
                result["bin_labels"] = self.bin_labels
                result["bin_edges"] = self.bin_edges
                result["bin_percentiles"] = getattr(self, "bin_percentiles", {})
                result["bin_value_ranges"] = getattr(self, "bin_value_ranges", [])
                result["bin_percentile_ranges"] = getattr(self, "bin_percentile_ranges", [])
                result["per_bin_sample_counts"] = self._per_bin_sample_counts
                result["per_bin_sample_proportions"] = self._per_bin_sample_proportions
            else:
                result["per_class_sample_counts"] = self._per_class_sample_counts
                result["per_class_sample_proportions"] = self._per_class_sample_proportions
                result["per_class_num_stats"] = self.per_class_num_stats
                result["per_class_cat_stats"] = self.per_class_cat_stats

        return result

    def save(self, path: str):
        """Save statistics to JSON file."""
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, indent=2, ensure_ascii=False)
        print(f"Saved statistics to {path}")
