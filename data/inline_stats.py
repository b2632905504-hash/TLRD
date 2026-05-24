"""
Inline (per-feature) statistics helpers.

These stats are intended to be injected directly after each feature value.
"""

from typing import Dict, List, Any

import numpy as np
import pandas as pd

from data.stats import ABLATION_VARIANTS


class InlineConditionalStats:
    """Conditional stats for inline feature descriptions."""

    def __init__(
        self,
        train_df: pd.DataFrame,
        num_cols: List[str],
        cat_cols: List[str],
        target_col: str,
        ablation_variant: str,
        task_type: str,
    ):
        self.train_df = train_df
        self.num_cols = num_cols
        self.cat_cols = cat_cols
        self.target_col = target_col
        self.ablation_variant = ablation_variant
        self.task_type = task_type

        if self.task_type == "regression":
            self._target_classes = []
            self._num_stats = self._build_regression_num_stats()
            self._cat_rates = self._build_regression_cat_rates()
            return

        self._target_classes = sorted(train_df[target_col].unique(), key=lambda x: str(x))
        self._num_stats = self._build_num_stats()
        self._cat_rates = self._build_cat_rates()

    def _build_regression_num_stats(self) -> Dict[str, Dict[str, float]]:
        stats_by_feature: Dict[str, Dict[str, float]] = {}
        for col in self.num_cols:
            s = self.train_df[col].dropna()
            if len(s) == 0:
                continue
            stats_by_feature[col] = {
                "mean": float(s.mean()),
                "std": float(s.std()),
                "min": float(s.min()),
                "max": float(s.max()),
                "p25": float(s.quantile(0.25)),
                "p50": float(s.quantile(0.50)),
                "p75": float(s.quantile(0.75)),
            }
        return stats_by_feature

    def _build_regression_cat_rates(self) -> Dict[str, Dict[str, float]]:
        rates_by_feature: Dict[str, Dict[str, float]] = {}
        for col in self.cat_cols:
            s = self.train_df[col].fillna("missing").replace("?", "missing")
            vc = s.value_counts(normalize=True, dropna=False)
            rates_by_feature[col] = {str(cat_val): float(rate) for cat_val, rate in vc.items()}
        return rates_by_feature

    def _build_num_stats(self) -> Dict[str, Dict[str, Dict[str, float]]]:
        stats_by_feature: Dict[str, Dict[str, Dict[str, float]]] = {}
        for col in self.num_cols:
            stats_by_feature[col] = {}
            for label in self._target_classes:
                s = self.train_df.loc[self.train_df[self.target_col] == label, col].dropna()
                if len(s) > 0:
                    stats = {
                        "mean": float(s.mean()),
                        "std": float(s.std()),
                        "min": float(s.min()),
                        "max": float(s.max()),
                        "p25": float(s.quantile(0.25)),
                        "p50": float(s.quantile(0.50)),
                        "p75": float(s.quantile(0.75)),
                    }
                else:
                    stats = {k: float("nan") for k in ["mean", "std", "min", "max", "p25", "p50", "p75"]}
                stats_by_feature[col][str(label)] = stats
        return stats_by_feature

    def _build_cat_rates(self) -> Dict[str, Dict[str, Dict[str, float]]]:
        rates_by_feature: Dict[str, Dict[str, Dict[str, float]]] = {}
        for col in self.cat_cols:
            per_class_rates: Dict[str, Dict[Any, float]] = {}
            all_categories = set()
            for label in self._target_classes:
                class_df = self.train_df[self.train_df[self.target_col] == label]
                s = class_df[col].fillna("missing").replace("?", "missing")
                vc = s.value_counts(normalize=True, dropna=False)
                per_class_rates[str(label)] = {cat_val: float(rate) for cat_val, rate in vc.items()}
                all_categories.update(vc.index.tolist())

            col_rates: Dict[str, Dict[str, float]] = {}
            for cat_val in sorted(all_categories, key=lambda x: str(x)):
                col_rates[str(cat_val)] = {
                    str(label): per_class_rates.get(str(label), {}).get(cat_val, 0.0)
                    for label in self._target_classes
                }
            rates_by_feature[col] = col_rates
        return rates_by_feature

    def format_feature_context(self, feature_name: str) -> str:
        """Return inline context string (without parentheses)."""
        if self.task_type == "regression":
            included_stats = ABLATION_VARIANTS[self.ablation_variant]
            if feature_name in self._num_stats:
                s = self._num_stats[feature_name]
                parts = []
                if "mean" in included_stats:
                    parts.append(f"mean={s['mean']:.2f}")
                if "std" in included_stats:
                    parts.append(f"std={s['std']:.2f}")
                if "p25" in included_stats:
                    parts.append(f"p25={s['p25']:.2f}")
                if "p50" in included_stats:
                    parts.append(f"p50={s['p50']:.2f}")
                if "p75" in included_stats:
                    parts.append(f"p75={s['p75']:.2f}")
                if "min" in included_stats:
                    parts.append(f"min={s['min']:.2f}")
                if "max" in included_stats:
                    parts.append(f"max={s['max']:.2f}")
                return f"train stats: {', '.join(parts)}"

            if feature_name in self._cat_rates:
                top3 = ", ".join(
                    f"{v}({p:.1%})" for v, p in list(self._cat_rates[feature_name].items())[:3]
                )
                return f"top categories: {top3}"
            return ""

        included_stats = ABLATION_VARIANTS[self.ablation_variant]

        if feature_name in self._num_stats:
            parts = []
            for label in self._target_classes:
                stats = self._num_stats[feature_name].get(str(label), {})
                stat_parts = []
                for key in included_stats:
                    val = stats.get(key, np.nan)
                    stat_parts.append(f"{key}={'NA' if np.isnan(val) else f'{val:.3f}'}")
                parts.append(f"when {self.target_col}={label}: {', '.join(stat_parts)}")
            return "; ".join(parts)

        if feature_name in self._cat_rates:
            cat_rates = self._cat_rates[feature_name]
            parts = []
            for label in self._target_classes:
                label_probs = []
                for cat_val, probs in cat_rates.items():
                    label_probs.append(f"{cat_val}={probs.get(str(label), 0.0):.3f}")
                parts.append(f"when {self.target_col}={label}: {', '.join(label_probs)}")
            return f"per-class proportions: {'; '.join(parts)}"

        return ""
