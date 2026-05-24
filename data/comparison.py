"""
RAG-style comparison module for tabular data prompts.

This module retrieves similar training samples for each test instance and
provides formatted text to include in prompts. Follows the same pattern as
stats.py - used by corpus_builder.py to enrich prompts.

Based on the retrieval logic from project_trm_llm/run_batch_prediction.py
"""

import sys
from pathlib import Path
from typing import Dict, Any, Optional, List, Tuple
from dataclasses import dataclass
import logging

import numpy as np
import pandas as pd

# Ensure local imports work when executed as a script
_SCRIPT_DIR = Path(__file__).resolve().parent
if str(_SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPT_DIR))
_PARENT_DIR = _SCRIPT_DIR.parent
if str(_PARENT_DIR) not in sys.path:
    sys.path.insert(0, str(_PARENT_DIR))

from data.dataset import TabularDataset

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


# Default categorical features configuration for known datasets
DEFAULT_CATEGORICAL_FEATURES = {
    "adult": {
        "categorical": [
            "marital-status", "native-country", "occupation", "education",
            "relationship", "workclass", "race", "gender",
        ]
    },
    "california": {
        "categorical": ["ocean_proximity"]
    },
}


@dataclass
class RetrievedExample:
    """Container for a single retrieved example with metadata."""
    index: int
    features: Dict[str, Any]
    target: Any
    target_label: str  # Human-readable target label from convert_target_to_text
    similarity: float
    prompt_text: str


class ComparisonExamples:
    """
    Compute and store similar training examples for RAG-style prompts.
    
    This class follows the same pattern as DatasetStats - it takes a training
    dataset and provides methods to retrieve and format similar examples for
    any given test sample.
    
    Used by corpus_builder.py to enrich prompts with similar historical cases.
    
    Usage:
        # Initialize with training dataset (like DatasetStats)
        comparison = ComparisonExamples(
            train_dataset=train_ds,
            n_examples=5,
            balance_rag=False,
        )
        
        # For each test sample, get formatted comparison text
        comparison_text = comparison.format_comparison_text(
            test_dataset=test_ds,
            test_idx=0,
            task="classification",
        )
    """
    
    # Instruction text prepended when comparison examples are included
    COMPARISON_INSTRUCTION = (
        "You are given several similar historical cases with their ground truth labels.\n"
        "Use them as guidance and then assess the current case.\n"
    )
    
    def __init__(
        self,
        train_dataset: TabularDataset,
        n_examples: int = 5,
        top_features: Optional[List[str]] = None,
        categorical_features: Optional[List[str]] = None,
        balance_rag: bool = False,
        use_random_rag: bool = False,
        include_similarity: bool = True,
        random_seed: int = 42,
    ):
        """
        Initialize comparison examples retriever with training data.
        
        Args:
            train_dataset: Training dataset to retrieve examples from
            n_examples: Number of similar examples to retrieve per query
            top_features: Feature columns to use for similarity computation
                         (defaults to all feature columns)
            categorical_features: Categorical features for exact-match filtering
            balance_rag: If True, retrieve class-balanced samples
                (binary: 50/50; multiclass: balanced across classes)
            use_random_rag: If True, use random sampling instead of similarity
            include_similarity: If True, include cosine similarity scores
            random_seed: Random seed for reproducibility
        """
        self.train_dataset = train_dataset
        self.n_examples = n_examples
        self.balance_rag = balance_rag
        self.use_random_rag = use_random_rag
        self.include_similarity = include_similarity
        self.random_seed = random_seed
        self.rng = np.random.default_rng(random_seed)
        
        self.train_df = train_dataset.df.copy()
        self.dataset_name = train_dataset.config.name
        self.n_train = len(train_dataset)
        
        # Determine feature columns
        if top_features is not None:
            self.top_features = [f for f in top_features if f in self.train_df.columns]
        else:
            self.top_features = train_dataset.feature_cols
        
        # Determine categorical features
        if categorical_features is not None:
            self.categorical_features = [f for f in categorical_features if f in self.top_features]
        else:
            dataset_name = train_dataset.config.name
            default_cat = DEFAULT_CATEGORICAL_FEATURES.get(dataset_name, {}).get("categorical", [])
            # Preserve deterministic priority: dataset-declared categories first,
            # then append default categories not already present.
            ordered_cat = list(dict.fromkeys((train_dataset.cat_cols or []) + (default_cat or [])))
            self.categorical_features = [f for f in ordered_cat if f in self.top_features]
        
        # Numeric features
        self.numeric_features = [f for f in self.top_features if f not in self.categorical_features]
        
        logger.info(f"ComparisonExamples initialized:")
        logger.info(f"  Training samples: {self.n_train}")
        logger.info(f"  Examples per query: {self.n_examples}")
        logger.info(f"  Numeric features: {len(self.numeric_features)}")
        logger.info(f"  Categorical features: {len(self.categorical_features)}")
        logger.info(f"  Balance RAG: {self.balance_rag}")
        
        # Precompute embeddings
        self._precompute_numeric_embeddings()
        
        # Cache for prompt texts (index -> (prompt_text, target_label))
        self._prompt_cache: Dict[int, Tuple[str, str]] = {}
    
    def _precompute_numeric_embeddings(self):
        """Precompute standardized and normalized numeric feature vectors."""
        if not self.numeric_features:
            self.rag_means = None
            self.rag_stds = None
            self.rag_num_unit = None
            return
        
        rag_num = self.train_df[self.numeric_features].astype(float)
        
        # Compute standardization stats
        self.rag_means = rag_num.mean(axis=0)
        self.rag_stds = rag_num.std(axis=0).replace(0.0, 1.0)
        
        # Standardize
        rag_num_std = (rag_num.fillna(self.rag_means) - self.rag_means) / self.rag_stds
        rag_num_matrix = rag_num_std.to_numpy(dtype=float)
        
        # Normalize for cosine similarity
        rag_norms = np.linalg.norm(rag_num_matrix, axis=1, keepdims=True)
        rag_norms[rag_norms == 0] = 1.0
        self.rag_num_unit = rag_num_matrix / rag_norms
    
    def _standardize_test_vector(self, test_features: Dict[str, Any]) -> np.ndarray:
        """Standardize a single test instance's numeric features."""
        if not self.numeric_features:
            return np.array([])
        
        values = []
        for col in self.numeric_features:
            val = test_features.get(col, np.nan)
            if val == "missing" or pd.isna(val):
                val = self.rag_means[col]
            values.append(float(val))
        
        test_vec = np.array(values)
        test_std = (test_vec - self.rag_means.values) / self.rag_stds.values
        
        norm = np.linalg.norm(test_std)
        if norm == 0:
            norm = 1.0
        
        return test_std / norm
    
    def _compute_similarities(
        self,
        test_features: Dict[str, Any],
        candidate_indices: np.ndarray,
    ) -> np.ndarray:
        """Compute cosine similarities between test instance and candidates."""
        if not self.numeric_features or self.rag_num_unit is None:
            return np.ones(len(candidate_indices))
        
        test_unit = self._standardize_test_vector(test_features)
        return self.rag_num_unit[candidate_indices] @ test_unit
    
    def _apply_categorical_filter(self, test_features: Dict[str, Any]) -> np.ndarray:
        """Filter candidates by exact match on categorical features.
        
        If applying a filter would reduce candidates to <= n_examples,
        we roll back to the previous mask to ensure enough samples.
        """
        candidate_idx = np.arange(len(self.train_df))
        
        if not self.categorical_features:
            return candidate_idx
        
        mask = np.ones(len(self.train_df), dtype=bool)
        
        for col in self.categorical_features:
            if col not in self.train_df.columns:
                continue
            
            val = test_features.get(col, np.nan)
            val_str = val if pd.notna(val) and val != "missing" else "NaN"
            rag_col_str = self.train_df[col].fillna("NaN").astype(str)
            
            new_mask = mask & (rag_col_str == str(val_str))
            
            # Roll back if applying this filter would leave too few samples
            if new_mask.sum() <= self.n_examples:
                break
            
            mask = new_mask
        
        if mask.any():
            return np.where(mask)[0]
        
        return candidate_idx
    
    def _convert_targets_to_binary(self, targets: np.ndarray) -> np.ndarray:
        """Convert targets to binary (0/1) for balanced retrieval."""
        if self.dataset_name == "adult":
            def to_binary(t):
                if isinstance(t, str):
                    t_lower = t.lower().strip()
                    return 1 if t_lower in ('>50k', 'yes', '1', 'true') else 0
                return 1 if t > 0 else 0
            return np.array([to_binary(t) for t in targets])
        else:
            # Default: median split for numeric, label encoding for categorical
            try:
                numeric_targets = np.array([float(t) for t in targets])
                median = np.median(numeric_targets)
                return (numeric_targets > median).astype(int)
            except (ValueError, TypeError):
                unique_vals = list(set(targets))
                if len(unique_vals) == 2:
                    return np.array([1 if t == unique_vals[1] else 0 for t in targets])
                else:
                    most_common = pd.Series(targets).mode()[0]
                    return np.array([1 if t == most_common else 0 for t in targets])
    
    def _retrieve_neighbors(
        self,
        test_features: Dict[str, Any],
        exclude_idx: Optional[int] = None,
    ) -> List[Tuple[int, float]]:
        """Retrieve similar training samples for a test instance."""
        if self.use_random_rag:
            candidate_idx = np.arange(len(self.train_df))
            if exclude_idx is not None:
                candidate_idx = candidate_idx[candidate_idx != exclude_idx]
            if len(candidate_idx) > self.n_examples:
                top_indices = self.rng.choice(candidate_idx, size=self.n_examples, replace=False)
            else:
                top_indices = candidate_idx
            return [(int(idx), 1.0) for idx in top_indices]
        
        # Apply categorical filter
        candidate_idx = self._apply_categorical_filter(test_features)
        
        if exclude_idx is not None:
            candidate_idx = candidate_idx[candidate_idx != exclude_idx]
        
        if len(candidate_idx) == 0:
            return []
        
        # Compute similarities
        sims = self._compute_similarities(test_features, candidate_idx)
        
        if self.balance_rag:
            return self._retrieve_balanced(candidate_idx, sims)
        else:
            return self._retrieve_top_n(candidate_idx, sims)
    
    def _retrieve_top_n(
        self,
        candidate_idx: np.ndarray,
        sims: np.ndarray,
    ) -> List[Tuple[int, float]]:
        """Retrieve top-n by similarity."""
        sorted_order = np.argsort(-sims)
        top_n = min(self.n_examples, len(sorted_order))
        
        results = []
        for i in range(top_n):
            idx = candidate_idx[sorted_order[i]]
            sim = sims[sorted_order[i]]
            results.append((int(idx), float(sim)))
        
        return results
    
    def _retrieve_balanced(
        self,
        candidate_idx: np.ndarray,
        sims: np.ndarray,
    ) -> List[Tuple[int, float]]:
        """Retrieve class-balanced samples.

        Binary datasets keep the original 50/50 logic.
        Multiclass datasets allocate near-equal quotas per class.
        """
        target_col = self.train_dataset.target_col
        targets = self.train_df.iloc[candidate_idx][target_col].values

        if self.train_dataset.task_type == "multiclass":
            return self._retrieve_balanced_multiclass(candidate_idx, sims, targets)

        binary_targets = self._convert_targets_to_binary(targets)
        
        pos_mask = binary_targets == 1
        neg_mask = binary_targets == 0
        
        pos_indices = candidate_idx[pos_mask]
        neg_indices = candidate_idx[neg_mask]
        pos_sims = sims[pos_mask]
        neg_sims = sims[neg_mask]
        
        n_per_class = self.n_examples // 2
        n_remaining = self.n_examples % 2
        
        results = []
        
        if len(pos_indices) > 0:
            pos_order = np.argsort(-pos_sims)
            n_pos = min(n_per_class, len(pos_order))
            for i in range(n_pos):
                results.append((int(pos_indices[pos_order[i]]), float(pos_sims[pos_order[i]])))
        
        if len(neg_indices) > 0:
            neg_order = np.argsort(-neg_sims)
            n_neg = min(n_per_class + n_remaining, len(neg_order))
            for i in range(n_neg):
                results.append((int(neg_indices[neg_order[i]]), float(neg_sims[neg_order[i]])))
        
        # Fill if underrepresented
        if len(results) < self.n_examples:
            selected_set = {r[0] for r in results}
            remaining_needed = self.n_examples - len(results)
            
            all_remaining = []
            for c_idx, sim in zip(candidate_idx, sims):
                if c_idx not in selected_set:
                    all_remaining.append((int(c_idx), float(sim)))
            
            all_remaining.sort(key=lambda x: -x[1])
            results.extend(all_remaining[:remaining_needed])
        
        results.sort(key=lambda x: -x[1])
        return results

    def _retrieve_balanced_multiclass(
        self,
        candidate_idx: np.ndarray,
        sims: np.ndarray,
        targets: np.ndarray,
    ) -> List[Tuple[int, float]]:
        """Retrieve balanced samples across all target classes."""
        # Factorize preserves first-seen class order and handles mixed dtypes safely.
        class_codes, class_labels = pd.factorize(pd.Series(targets), sort=False)
        valid_mask = class_codes >= 0
        if not np.any(valid_mask):
            return self._retrieve_top_n(candidate_idx, sims)

        candidate_idx = candidate_idx[valid_mask]
        sims = sims[valid_mask]
        class_codes = class_codes[valid_mask]

        n_classes = len(class_labels)
        if n_classes <= 1:
            return self._retrieve_top_n(candidate_idx, sims)

        n_per_class = self.n_examples // n_classes
        n_remaining = self.n_examples % n_classes

        class_to_rows: Dict[int, List[Tuple[int, float]]] = {}
        for class_id in range(n_classes):
            cls_mask = class_codes == class_id
            cls_indices = candidate_idx[cls_mask]
            cls_sims = sims[cls_mask]
            cls_order = np.argsort(-cls_sims)
            class_to_rows[class_id] = [
                (int(cls_indices[i]), float(cls_sims[i])) for i in cls_order
            ]

        results: List[Tuple[int, float]] = []
        selected: set[int] = set()

        # First pass: equal quota per class.
        for class_id in range(n_classes):
            rows = class_to_rows[class_id]
            n_take = min(n_per_class, len(rows))
            for idx, sim in rows[:n_take]:
                results.append((idx, sim))
                selected.add(idx)
            class_to_rows[class_id] = rows[n_take:]

        # Second pass: allocate remainder to classes with the strongest next candidate.
        if n_remaining > 0:
            class_priority = sorted(
                range(n_classes),
                key=lambda c: class_to_rows[c][0][1] if class_to_rows[c] else float("-inf"),
                reverse=True,
            )
            for class_id in class_priority:
                if n_remaining <= 0:
                    break
                rows = class_to_rows[class_id]
                if not rows:
                    continue
                idx, sim = rows.pop(0)
                if idx not in selected:
                    results.append((idx, sim))
                    selected.add(idx)
                    n_remaining -= 1

        # Final fill: backfill by global similarity if some classes are underrepresented.
        if len(results) < self.n_examples:
            all_remaining: List[Tuple[int, float]] = []
            for rows in class_to_rows.values():
                for idx, sim in rows:
                    if idx not in selected:
                        all_remaining.append((idx, sim))
            all_remaining.sort(key=lambda x: -x[1])
            results.extend(all_remaining[: self.n_examples - len(results)])

        results.sort(key=lambda x: -x[1])
        return results
    
    def _get_target_label(self, target: Any, task: Optional[str] = None) -> str:
        """Get human-readable target label using dataset-specific convert_target_to_text."""
        try:
            if self.dataset_name == "adult":
                from data.adult.adult_dataset import convert_target_to_text
                return convert_target_to_text(target)
            elif self.dataset_name == "california":
                from data.california.california_dataset import convert_target_to_text
                return convert_target_to_text(target, task=task)
            elif self.dataset_name == "diamonds":
                from data.diamonds.diamonds_dataset import convert_target_to_text
                return convert_target_to_text(target, task=task)
            else:
                return self._default_target_label(target, task)
        except ImportError as e:
            logger.warning(f"Could not import convert_target_to_text for {self.dataset_name}: {e}")
            return self._default_target_label(target, task)
    
    def _default_target_label(self, target: Any, task: Optional[str] = None) -> str:
        """Default fallback for target labels."""
        if isinstance(target, (int, float)):
            if task == "classification":
                return f"The label for this case is: {target}"
            else:
                return f"The target value is: {target}"
        return f"The label is: {target}"
    
    def _get_example_prompt_text(self, train_idx: int, task: Optional[str] = None) -> str:
        """Get the prompt text for a training example."""
        if hasattr(self.train_dataset, "build_prompt"):
            try:
                if task and self.dataset_name == "california":
                    prompt_dict = self.train_dataset.build_prompt(train_idx, task=task)
                else:
                    prompt_dict = self.train_dataset.build_prompt(train_idx)
                return prompt_dict.get("input", "")
            except Exception:
                pass
        
        # Fallback: format features as text
        sample = self.train_dataset[train_idx]
        features = sample["features"]
        lines = []
        for col, val in features.items():
            if val == "missing" or (isinstance(val, float) and pd.isna(val)):
                val_str = "missing"
            else:
                val_str = str(val)
            lines.append(f"{col}: {val_str}")
        return ", ".join(lines)
    
    def format_comparison_text(
        self,
        test_features: Dict[str, Any],
        task: Optional[str] = None,
        include_header: bool = True,
        mode: str = "zeroshot",
        exclude_idx: Optional[int] = None,
    ) -> str:
        """
        Format comparison examples as text for a given test instance.
        
        This is the main method called by corpus_builder.py, similar to
        DatasetStats.format_stats_text().
        
        Args:
            test_features: Dictionary of feature values for the test instance
            task: Optional task type (e.g., "classification", "regression")
            include_header: Whether to include section header
            mode: Corpus builder mode used for display details.
            exclude_idx: Optional index to exclude from retrieval (e.g., self in train split)
            
        Returns:
            Formatted comparison text with similar examples
        """
        # Retrieve similar neighbors
        neighbors = self._retrieve_neighbors(test_features, exclude_idx=exclude_idx)
        
        if not neighbors:
            return ""
        
        lines = []
        
        if include_header:
            lines.append("=== Similar Historical Cases ===")

        for rank, (train_idx, similarity) in enumerate(neighbors, start=1):
            # Get cached or compute prompt text and target label
            if train_idx not in self._prompt_cache:
                prompt_text = self._get_example_prompt_text(train_idx, task)
                sample = self.train_dataset[train_idx]
                target_label = self._get_target_label(sample["target"], task)
                self._prompt_cache[train_idx] = (prompt_text, target_label)
            else:
                prompt_text, target_label = self._prompt_cache[train_idx]

            # Format example
            if self.include_similarity:
                if mode == "finetune":
                    lines.append(f"(similarity: {similarity:.3f}):")
                else:
                    lines.append(f"Example {rank} (similarity: {similarity:.3f}):")
            else:
                lines.append(f"Example {rank}:")

            lines.append(f"  Features: {prompt_text}")
            lines.append(f"  Ground truth: {target_label}")
        
        return "\n".join(lines)
    
    def to_dict(self) -> Dict[str, Any]:
        """Export configuration as a dictionary."""
        return {
            "n_train": self.n_train,
            "n_examples": self.n_examples,
            "dataset_name": self.dataset_name,
            "balance_rag": self.balance_rag,
            "use_random_rag": self.use_random_rag,
            "include_similarity": self.include_similarity,
            "numeric_features": self.numeric_features,
            "categorical_features": self.categorical_features,
        }


# Example usage and testing
if __name__ == "__main__":
    from data.dataset_registry import get_dataset_class
    
    print("Testing ComparisonExamples with Adult dataset...")
    
    AdultDataset = get_dataset_class("adult")
    train_ds = AdultDataset(split="train", seed=42)
    test_ds = AdultDataset(split="test", seed=42)
    
    # Initialize comparison (like DatasetStats)
    comparison = ComparisonExamples(
        train_dataset=train_ds,
        n_examples=3,
        balance_rag=False,
        include_similarity=True,
    )
    
    # Get test sample features
    test_sample = test_ds[0]
    test_features = test_sample["features"]
    
    # Format comparison text (like format_stats_text)
    comparison_text = comparison.format_comparison_text(test_features)
    
    print("\n" + "=" * 80)
    print("Sample comparison text:")
    print("=" * 80)
    print(comparison_text)
    
    # Test with balanced retrieval
    print("\n" + "=" * 80)
    print("Testing balanced RAG retrieval...")
    print("=" * 80)
    
    comparison_balanced = ComparisonExamples(
        train_dataset=train_ds,
        n_examples=10,
        balance_rag=True,
        include_similarity=True,
    )
    
    comparison_text_balanced = comparison_balanced.format_comparison_text(test_features)
    print(comparison_text_balanced)
