"""
Corpus Builder for Tabular Datasets

Builds Alpaca-style JSON training corpus from TabularDataset subclasses.
Supports building corpus for specific splits (train/valid/test).
Optionally enriches prompts with dataset statistics (computed from train set only).

Usage:
    # Build train corpus for adult dataset
    python corpus_builder.py --dataset adult --split train --seed 42
    
    # Build all splits
    python corpus_builder.py --dataset adult --split all --seed 42
    
    # Limit number of samples
    python corpus_builder.py --dataset adult --split train --limit 10000
    
    # Build with statistics (stats computed from train set)
    python corpus_builder.py --dataset adult --split train --with-stats
"""

import argparse
import json
import os
import sys
import shlex
from datetime import datetime
from pathlib import Path
from typing import Optional, List, Literal, Dict, Any, Union

import numpy as np
import pandas as pd

# Add parent directory to path for consistent imports
_SCRIPT_DIR = Path(__file__).resolve().parent
if str(_SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPT_DIR))
# Also add parent of data/ so "data.xxx" imports work
_PARENT_DIR = _SCRIPT_DIR.parent
if str(_PARENT_DIR) not in sys.path:
    sys.path.insert(0, str(_PARENT_DIR))

# Dataset imports
from dataset import TabularDataset, DATASET_REGISTRY
from data.dataset_registry import get_dataset_class
from data.stats import DatasetStats, ABLATION_VARIANTS
from data.inline_stats import InlineConditionalStats
from data.comparison import ComparisonExamples


# ==================== Corpus Builder ====================

class CorpusBuilder:
    """
    Builds Alpaca-style JSON corpus from TabularDataset.
    
    Output format:
    [
        {
            "instruction": "...",
            "input": "feature1 is value1. feature2 is value2. ...",
            "output": "target_col is target_value"
        },
        ...
    ]
    
    With stats enabled, the input is enriched with dataset statistics
    computed from the training set (no data leakage).
    
    With comparison enabled, the input includes similar historical cases
    retrieved from the training set (RAG-style examples).
    """
    
    # Contextual suffixes to append to original instruction
    # These describe what additional information is provided for reasoning
    STATS_ONLY_SUFFIX = (
        "You are also provided with dataset statistics to assist your reasoning."
    )
    
    COMPARISON_ONLY_SUFFIX = (
        "You are also provided with similar historical cases to assist your reasoning."
    )
    
    STATS_COMPARISON_SUFFIX = (
        "You are also provided with dataset statistics and similar historical cases to assist your reasoning."
    )
    
    def __init__(
        self,
        dataset_name: str,
        seed: int = 42,
        output_dir: Optional[str] = None,
        run_command: Optional[str] = None,
        with_stats: bool = False,
        inline_feature_stats: bool = False,
        inline_header_block: bool = False,
        include_stats_header_block: bool = True,
        include_stats_block: bool = True,
        stats_top_k: Union[int, str, None] = 5,
        stats_position: Literal["prefix", "suffix"] = "prefix",
        ablation_variant: str = "full",
        per_class_stats: bool = False,
        task: Optional[Literal["regression", "classification"]] = None,
        # Comparison options
        with_comparison: bool = False,
        comparison_n: int = 5,
        comparison_position: Literal["before_query", "after_query"] = "before_query",
        balance_comparison: bool = False,
        include_similarity: bool = True,
        # Mode options
        mode: Literal["zeroshot", "reasoning", "finetune", "normal_finetune"] = "zeroshot",
        # Sampling options
        undersampling: bool = False,
        # Output options
        no_timestamp: Optional[str] = None,
    ):
        """
        Args:
            dataset_name: Name of the registered dataset
            seed: Random seed for split files
            output_dir: Directory to save corpus files (default: dataset dir/corpus)
            with_stats: Whether to include dataset statistics in prompts
            inline_feature_stats: Inline per-feature stats in the sample input
            inline_header_block: Add header + target distribution before inline features
            include_stats_block: Whether to include the full stats block text
            stats_top_k: Number of top categories to include for categorical features
                (or \"all\" for all categories; default: 5)
            stats_position: Where to add stats in input ("prefix" or "suffix")
            ablation_variant: Which numerical stats to include.
                Options: "mean_only", "mean_std", "no_minmax", "full"
            per_class_stats: Whether to compute stats separately for each target class
            task: Task type for datasets that support multiple tasks
            with_comparison: Whether to include similar historical cases in prompts
            comparison_n: Number of similar examples to retrieve
            comparison_position: Where to add comparison relative to query data
                "before_query": comparison examples appear before sample features
                "after_query": comparison examples appear after sample features
            balance_comparison: If True, retrieve 50% positive / 50% negative samples
            include_similarity: If True, include cosine similarity scores in examples
            mode: Corpus generation mode:
                - "zeroshot": Standard prediction mode
                - "reasoning": Use dataset-specific instruction_reasoning and skip suffixes
                - "finetune": Use finetune instruction with given prediction label
                - "normal_finetune": Use INSTRUCTION_NORMAL_FINETUNE with given prediction label
            undersampling: In zeroshot/finetune/normal_finetune mode with limit, sample in repeated order:
                negative, negative, negative, positive (3:1 pattern).
            no_timestamp:
                Controls the output subdirectory behavior (replaces the old boolean flag):
                - None (default): create a timestamp subdirectory (original behavior when --no-timestamp is NOT provided)
                - "" (empty string): do NOT create any subdirectory; save directly under base output directory
                  (original behavior when --no-timestamp is provided as a flag)
                - any other string: create a subdirectory with this string under base output directory
        """
        self.dataset_name = dataset_name
        self.seed = seed
        self.dataset_class = get_dataset_class(dataset_name)
        self.inline_feature_stats = inline_feature_stats
        self.inline_header_block = inline_header_block
        self.include_stats_header_block = include_stats_header_block
        self.include_stats_block = include_stats_block
        self.with_stats = with_stats or inline_feature_stats
        self.stats_top_k = stats_top_k
        self.stats_position = stats_position
        self.ablation_variant = ablation_variant
        self.per_class_stats = per_class_stats
        # Optional task type for datasets that support multiple tasks
        # (e.g., California housing: regression vs classification).
        self.task = task
        
        # Comparison options
        self.with_comparison = with_comparison
        self.comparison_n = comparison_n
        self.comparison_position = comparison_position
        self.balance_comparison = balance_comparison
        self.include_similarity = include_similarity
        
        # Mode options
        self.mode = mode
        self.undersampling = undersampling
        
        # Determine output directory
        if output_dir is None:
            if dataset_name in DATASET_REGISTRY:
                config = DATASET_REGISTRY[dataset_name]
                data_dir = Path(config.csv_path).parent
                output_dir = str(data_dir / "corpus")
            else:
                output_dir = f"corpus/{dataset_name}"
        
        self.base_output_dir = output_dir
        # Output directory selection:
        # - no_timestamp is None: use timestamp subdir (default)
        # - no_timestamp is "": save directly to base dir
        # - no_timestamp is "some_str": save to base_dir/some_str
        if no_timestamp is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            self.output_dir = os.path.join(self.base_output_dir, timestamp)
        elif no_timestamp == "":
            self.output_dir = self.base_output_dir
        else:
            self.output_dir = os.path.join(self.base_output_dir, no_timestamp)
        os.makedirs(self.output_dir, exist_ok=True)

        if run_command:
            command_path = os.path.join(self.output_dir, "command.txt")
            with open(command_path, "w", encoding="utf-8") as f:
                f.write(run_command.strip() + "\n")
        
        # Initialize statistics (computed lazily from train set)
        self._stats: Optional[DatasetStats] = None

        # Initialize inline conditional stats (computed lazily from train set)
        self._inline_conditional_stats: Optional[InlineConditionalStats] = None
        
        # Initialize comparison examples (computed lazily from train set)
        self._comparison: Optional[ComparisonExamples] = None

    def _resolve_instruction_reasoning(
        self,
        prompt: Optional[dict],
        dataset: Optional[TabularDataset],
    ) -> Optional[str]:
        if prompt is not None:
            prompt_instruction = prompt.get("instruction_reasoning")
            if isinstance(prompt_instruction, str) and prompt_instruction.strip():
                return prompt_instruction

        if dataset is not None and hasattr(dataset, "INSTRUCTION_REASONING"):
            return dataset.INSTRUCTION_REASONING
        if hasattr(self.dataset_class, "INSTRUCTION_REASONING"):
            return self.dataset_class.INSTRUCTION_REASONING

        task_attr = None
        if self.task:
            task_attr = f"INSTRUCTION_REASONING_{self.task.upper()}"
            if dataset is not None and hasattr(dataset, task_attr):
                return getattr(dataset, task_attr)
            if hasattr(self.dataset_class, task_attr):
                return getattr(self.dataset_class, task_attr)

        candidate_attrs = [
            name for name in dir(self.dataset_class)
            if name.startswith("INSTRUCTION_REASONING_")
        ]
        if len(candidate_attrs) == 1:
            return getattr(self.dataset_class, candidate_attrs[0])

        raise AttributeError(
            "No reasoning instruction found. Expected 'instruction_reasoning' in prompt, "
            "'INSTRUCTION_REASONING', or a task-specific attribute like "
            f"'{task_attr or 'INSTRUCTION_REASONING_<TASK>'}'."
        )

    def _resolve_instruction_finetune(self) -> Optional[str]:
        # if hasattr(self.dataset_class, "INSTRUCTION_FIENTUNE"):
        #     return self.dataset_class.INSTRUCTION_FIENTUNE
        # if hasattr(self.dataset_class, "FINETUNE_INSTRUCTION"):
        #     return self.dataset_class.FINETUNE_INSTRUCTION
        # return None
        assert hasattr(self.dataset_class, "INSTRUCTION_FINETUNE")  # Since there are many modes now, we need to assert the instruction is present, or the output might be confusing.
        return self.dataset_class.INSTRUCTION_FINETUNE

    def _resolve_instruction_normal_finetune(self) -> Optional[str]:
        assert hasattr(
            self.dataset_class, "INSTRUCTION_NORMAL_FINETUNE"
        )  # Keep the failure explicit so prompt selection remains predictable.
        return self.dataset_class.INSTRUCTION_NORMAL_FINETUNE

    @staticmethod
    def _coerce_binary_zero_one(value: Any) -> Optional[int]:
        """Map a value to 0/1 when possible, otherwise return None."""
        if isinstance(value, (bool, np.bool_)):
            return int(value)
        if isinstance(value, (int, np.integer)):
            v = int(value)
            if v in (0, 1):
                return v
            return None
        if isinstance(value, (float, np.floating)):
            if np.isnan(value):
                return None
            v = float(value)
            if abs(v) < 1e-12:
                return 0
            if abs(v - 1.0) < 1e-12:
                return 1
            return None
        s = str(value).strip().lower()
        if s in ("0", "0.0"):
            return 0
        if s in ("1", "1.0"):
            return 1
        return None

    def _resolve_binary_labels_for_undersampling(
        self,
        targets: np.ndarray,
    ) -> tuple[Any, Any, bool]:
        """
        Resolve (negative_label, positive_label, used_fallback).
        Prefer explicit 0/1 labels; otherwise fallback to majority/minority labels.
        """
        unique_labels = list(pd.unique(targets))
        if len(unique_labels) != 2:
            raise ValueError(
                f"--undersampling requires binary labels, got {len(unique_labels)} classes: {unique_labels}"
            )

        by_binary_value: Dict[int, Any] = {}
        for label in unique_labels:
            mapped = self._coerce_binary_zero_one(label)
            if mapped is not None and mapped not in by_binary_value:
                by_binary_value[mapped] = label

        if 0 in by_binary_value and 1 in by_binary_value:
            return by_binary_value[0], by_binary_value[1], False

        # Fallback when labels are not explicit 0/1 (e.g., yes/no): use majority as negative.
        value_counts = pd.Series(targets).value_counts()
        negative_label = value_counts.index[0]
        positive_label = value_counts.index[-1]
        return negative_label, positive_label, True

    def _build_undersampled_indices(
        self,
        targets: np.ndarray,
        n_samples: int,
    ) -> List[int]:
        """
        Build indices in repeated order: negative, negative, negative, positive (3:1).
        Sampling is without replacement, with class-internal shuffle controlled by self.seed.
        """
        negative_label, positive_label, used_fallback = self._resolve_binary_labels_for_undersampling(targets)
        rng = np.random.default_rng(self.seed)

        negative_indices = np.where(targets == negative_label)[0]
        positive_indices = np.where(targets == positive_label)[0]
        negative_indices = rng.permutation(negative_indices).tolist()
        positive_indices = rng.permutation(positive_indices).tolist()

        selected: List[int] = []
        neg_ptr, pos_ptr = 0, 0
        for i in range(n_samples):
            need_positive = (i % 4) == 3
            if need_positive:
                if pos_ptr >= len(positive_indices):
                    break
                selected.append(int(positive_indices[pos_ptr]))
                pos_ptr += 1
            else:
                if neg_ptr >= len(negative_indices):
                    break
                selected.append(int(negative_indices[neg_ptr]))
                neg_ptr += 1

        if used_fallback:
            print(
                "Warning: --undersampling could not find explicit 0/1 labels; "
                f"using majority label={negative_label} as negative and minority label={positive_label} as positive."
            )
        print(
            f"Undersampling pattern [neg, neg, neg, pos] with labels "
            f"[{negative_label}, {negative_label}, {negative_label}, {positive_label}]"
        )
        if len(selected) < n_samples:
            print(
                "Warning: undersampling stopped early due class exhaustion. "
                f"Requested {n_samples}, got {len(selected)} "
                f"(available negatives={len(negative_indices)}, positives={len(positive_indices)})."
            )

        return selected
    
    
    def _load_dataset(
        self,
        split: Optional[Literal["train", "valid", "test"]] = None,
    ) -> TabularDataset:
        """Load dataset with the specified split"""

        return self.dataset_class(split=split, seed=self.seed)
    
    def _get_stats(self) -> DatasetStats:
        """
        Get or compute dataset statistics from training set.
        Statistics are computed lazily and cached.
        """
        if self._stats is None:
            print("Computing statistics from training set...")
            train_dataset = self._load_dataset(split="train")
            
            
            if self.stats_top_k is None:
                stats_top_k = None
            elif isinstance(self.stats_top_k, str):
                if self.stats_top_k.lower() == "all":
                    stats_top_k = None
                else:
                    stats_top_k = int(self.stats_top_k)
            else:
                stats_top_k = int(self.stats_top_k)

            self._stats = DatasetStats(
                train_dataset,
                top_k=stats_top_k,
                ablation_variant=self.ablation_variant,
                per_class=self.per_class_stats,
            )
            print(f"  - {len(train_dataset)} training samples")
            print(f"  - {len(self._stats.num_stats)} numerical features")
            print(f"  - {len(self._stats.cat_stats)} categorical features")
            print(f"  - ablation_variant: {self.ablation_variant}")
            print(f"  - included stats: {ABLATION_VARIANTS[self.ablation_variant]}")
            print(f"  - per_class_stats: {self.per_class_stats}")
            if self.per_class_stats:
                if self._stats.task_type == "regression":
                    print(f"  - regression per-bin mode: {len(self._stats.bin_labels)} bins")
                    print(f"  - bin edges: {self._stats.bin_edges}")
                else:
                    print(f"  - target classes: {self._stats.target_classes}")
        return self._stats
    
    def _get_comparison(self) -> ComparisonExamples:
        """
        Get or compute comparison examples retriever from training set.
        Computed lazily and cached.
        """
        if self._comparison is None:
            print("Initializing comparison examples from training set...")
            train_dataset = self._load_dataset(split="train")
            
            self._comparison = ComparisonExamples(
                train_dataset=train_dataset,
                n_examples=self.comparison_n,
                balance_rag=self.balance_comparison,
                include_similarity=self.include_similarity,
                random_seed=self.seed,
            )
            print(f"  - {self._comparison.n_train} training samples")
            print(f"  - {self.comparison_n} examples per query")
            print(f"  - balance_comparison: {self.balance_comparison}")
            print(f"  - comparison_position: {self.comparison_position}")
        return self._comparison

    def _get_inline_conditional_stats(self, train_dataset: TabularDataset) -> InlineConditionalStats:
        if self._inline_conditional_stats is None:
            self._inline_conditional_stats = InlineConditionalStats(
                train_df=train_dataset.df,
                num_cols=train_dataset.num_cols,
                cat_cols=train_dataset.cat_cols,
                target_col=train_dataset.target_col,
                ablation_variant=self.ablation_variant,
                task_type=train_dataset.task_type,
            )
        return self._inline_conditional_stats

    def _inline_stats_into_input(
        self,
        stats: DatasetStats,
        dataset: TabularDataset,
        test_features: Dict[str, Any],
    ) -> str:
        """
        Inline feature statistics into each "Feature is value" entry.

        Example:
            "Age is 39" -> "Age is 39 (train stats: mean=..., p25=...)"
        """
        lines = []
        cond_stats = self._get_inline_conditional_stats(dataset)
        for col in dataset.feature_cols:
            value = test_features.get(col)
            if value == "missing" or (isinstance(value, float) and str(value) == "nan"):
                val_str = "missing"
            elif value == "?":
                val_str = "missing"
            else:
                val_str = str(value)

            context = cond_stats.format_feature_context(col)

            if context:
                lines.append(f"{col} is {val_str} ({context})")
            else:
                lines.append(f"{col} is {val_str}")

        return ". ".join(lines)
    
    def _enrich_prompt(
        self,
        prompt: dict,
        stats: Optional[DatasetStats],
        comparison: Optional[ComparisonExamples],
        test_features: Dict[str, Any],
        dataset: Optional[TabularDataset] = None,
        given_label: Optional[str] = None,
        exclude_idx: Optional[int] = None,
    ) -> dict:
        """
        Enrich a prompt with dataset statistics and/or comparison examples.
        
        The order of components in the enriched input:
        1. Statistics (if enabled) - always first as context
        2. Comparison examples (if enabled) - after stats
           - Position relative to query data controlled by comparison_position
        3. Sample features (query data)
        
        The original instruction from the dataset is preserved. A contextual
        prefix is prepended to indicate what additional information is provided.
        
        Args:
            prompt: Original prompt dict with instruction, input, output
            stats: DatasetStats object (or None if not enabled)
            comparison: ComparisonExamples object (or None if not enabled)
            test_features: Feature dictionary for the current sample
            
        Returns:
            Enriched prompt dict
        """
        parts = []
        query_text = prompt['input']
        original_instruction = prompt.get('instruction', '')

        if given_label:
            if query_text.endswith("\n"):
                query_text = query_text + given_label
            else:
                query_text = query_text + "\n\n" + given_label
        if self.mode == "reasoning":
            instruction_reasoning = self._resolve_instruction_reasoning(prompt, dataset)
            if instruction_reasoning:
                original_instruction = instruction_reasoning

        header_text = ""
        # Add statistics text (always first if enabled)
        if stats is not None and self.inline_feature_stats and dataset is not None:
            query_text = self._inline_stats_into_input(
                stats=stats,
                dataset=dataset,
                test_features=test_features,
            )
            if self.inline_header_block and not self.include_stats_block and self.include_stats_header_block:
                header_lines = []
                if stats.per_class and stats.task_type == "regression":
                    stats_mode = "per-bin"
                elif stats.per_class:
                    stats_mode = "per-class"
                else:
                    stats_mode = "overall"
                header_lines.append(
                    f"=== Dataset Statistics (from training set, {stats_mode}) ==="
                )
                header_lines.append(f"Total training samples: {stats.n_train}")
                if stats.target_dist:
                    target_name = getattr(stats, "target_display_name", stats.target_col)
                    header_lines.append(f"\nTarget distribution ({target_name}):")
                    for label, count in stats.target_dist.items():
                        prop = stats.target_proportions[label]
                        label_display = stats.display_target_label(label)
                        header_lines.append(f"  {label_display}: {count} ({prop:.1%})")
                header_text = "\n".join(header_lines)

        if stats is not None and self.include_stats_block:
            stats_text = stats.format_stats_text(
                include_header=self.include_stats_header_block,
                current_sample_features=test_features,
            )
            parts.append(stats_text)
        elif header_text:
            parts.append(header_text)
        
        # Get comparison text if enabled
        comparison_text = None
        if comparison is not None:
            comparison_text = comparison.format_comparison_text(
                test_features=test_features,
                task=self.task,
                include_header=True,
                mode=self.mode,
                exclude_idx=exclude_idx,
            )
        
        # Build enriched input based on comparison_position
        if comparison_text:
            if self.comparison_position == "before_query":
                # Stats -> Comparison -> Query
                parts.append(comparison_text)
                parts.append(f"--- Current Sample Features ---\n{query_text}")
            else:  # after_query
                # Stats -> Query -> Comparison
                parts.append(f"--- Current Sample Features ---\n{query_text}")
                parts.append(comparison_text)
        else:
            # No comparison, just add query
            if parts:  # Has stats
                parts.append(f"--- Sample Features ---\n{query_text}")
            else:
                parts.append(query_text)
        
        enriched_input = "\n\n".join(parts)
        
        # Determine instruction suffix (appended to original instruction)
        # The original dataset instruction is preserved to maintain task definition
        if self.mode == "zeroshot":
            if stats is not None and comparison is not None:
                suffix = self.STATS_COMPARISON_SUFFIX
            elif stats is not None:
                suffix = self.STATS_ONLY_SUFFIX
            elif comparison is not None:
                suffix = self.COMPARISON_ONLY_SUFFIX
            else:
                suffix = ""
        else:
            suffix = ""
        
        # Combine original instruction with suffix
        if suffix:
            enriched_instruction = original_instruction.rstrip() + " " + suffix
        else:
            enriched_instruction = original_instruction
        
        return {
            "instruction": enriched_instruction,
            "input": enriched_input,
            "output": prompt["output"],
        }
    
    def build_corpus(
        self,
        split: Literal["train", "valid", "test"],
        limit: Optional[int] = None,
        output_name: Optional[str] = None,
    ) -> str:
        """
        Build corpus for a specific split.
        
        Args:
            split: Which split to build corpus for
            limit: Maximum number of samples (None for all)
            output_name: Custom output filename (default: {dataset}_{split}_seed{seed}.json)
            
        Returns:
            Path to the saved corpus file
        """
        # Load dataset
        dataset = self._load_dataset(split)
        print(f"Loaded {self.dataset_name} {split} split: {len(dataset)} samples")
        
        # Get statistics if enabled (always from train set)
        stats = self._get_stats() if self.with_stats else None
        
        # Get comparison examples retriever if enabled (always from train set)
        comparison = self._get_comparison() if self.with_comparison else None
        
        # Determine number of samples and sampling indices
        total_samples = len(dataset)
        sample_indices = list(range(total_samples))
        
        if limit is not None and limit > 0:
            n_samples = min(total_samples, limit)
            if self.undersampling and self.mode in {"zeroshot", "finetune", "normal_finetune"}:
                if dataset.config.task_type != "binary":
                    raise ValueError(
                        "--undersampling currently only supports binary classification datasets in zeroshot/finetune/normal_finetune mode."
                    )
                targets = dataset.get_targets()
                sample_indices = self._build_undersampled_indices(targets, n_samples)
                print(f"Undersampling selected {len(sample_indices)} samples")
            elif dataset.config.task_type in ("binary", "multiclass"):
                rng = np.random.default_rng(self.seed)
                targets = dataset.get_targets()
                unique_labels, counts = np.unique(targets, return_counts=True)
                proportions = counts / counts.sum()
                raw_counts = proportions * n_samples
                per_class = np.floor(raw_counts).astype(int)
                remainder = n_samples - per_class.sum()

                if remainder > 0:
                    fractional = raw_counts - per_class
                    order = np.argsort(-fractional)
                    for idx in order[:remainder]:
                        per_class[idx] += 1

                selected = []
                for label, k in zip(unique_labels, per_class):
                    label_indices = np.where(targets == label)[0]
                    if k > len(label_indices):
                        k = len(label_indices)
                    if k > 0:
                        selected.extend(rng.choice(label_indices, size=k, replace=False).tolist())

                rng.shuffle(selected)
                sample_indices = selected
                print(f"Stratified sampling to {n_samples} samples")
            else:
                sample_indices = sample_indices[:n_samples]
                print(f"Limiting to {n_samples} samples")
        n_samples = len(sample_indices)
        
        # Build prompts
        samples = []
        for i in range(n_samples):
            # Get sample features for comparison retrieval
            dataset_idx = sample_indices[i]
            sample = dataset[dataset_idx]
            test_features = sample["features"]
            
            # Some dataset classes (e.g., CaliforniaDataset) support
            # multiple tasks through a `task` argument in build_prompt,
            # while others (e.g., AdultDataset) do not. We try to pass
            # `self.task` when provided and silently fall back if the
            # signature does not accept it.
            if self.task is not None:
                try:
                    prompt = dataset.build_prompt(dataset_idx, task=self.task)
                except TypeError:
                    prompt = dataset.build_prompt(dataset_idx)
            else:
                prompt = dataset.build_prompt(dataset_idx)

            given_label = None
            if self.mode in {"finetune", "normal_finetune"}:
                if self.mode == "finetune":
                    finetune_instruction = self._resolve_instruction_finetune()
                else:
                    finetune_instruction = self._resolve_instruction_normal_finetune()
                if finetune_instruction:
                    if hasattr(dataset, "_build_feature_explanations"):
                        try:
                            feature_explanations = dataset._build_feature_explanations()
                        except Exception:
                            feature_explanations = ""
                        if feature_explanations:
                            finetune_instruction = (
                                finetune_instruction.rstrip()
                                + "\n\n"
                                + feature_explanations
                                + "\n"
                            )
                    prompt["instruction"] = finetune_instruction

                output_label = str(prompt.get("output", "")).strip()
                if output_label:
                    given_label = f"THE GIVEN PREDICTION LABEL is {output_label}."

            if self.mode == "reasoning":
                instruction_reasoning = self._resolve_instruction_reasoning(prompt, dataset)
                if instruction_reasoning:
                    if hasattr(dataset, "_build_feature_explanations"):
                        try:
                            feature_explanations = dataset._build_feature_explanations()
                        except Exception:
                            feature_explanations = ""
                        if feature_explanations and "Feature explanations" not in instruction_reasoning:
                            instruction_reasoning = (
                                instruction_reasoning.rstrip()
                                + "\n\n"
                                + feature_explanations
                                + "\n"
                            )
                    prompt["instruction"] = instruction_reasoning
            if self.mode != "reasoning":
                prompt.pop("instruction_reasoning", None)
            
            # Enrich with statistics and/or comparison if enabled
            if stats is not None or comparison is not None:
                prompt = self._enrich_prompt(
                    prompt,
                    stats,
                    comparison,
                    test_features,
                    dataset=dataset,
                    given_label=given_label,
                    exclude_idx=dataset_idx if split == "train" else None,
                )
            elif given_label:
                if prompt["input"].endswith("\n"):
                    prompt["input"] = prompt["input"] + given_label
                else:
                    prompt["input"] = prompt["input"] + "\n\n" + given_label
            
            samples.append(prompt)
            
            # Progress
            if (i + 1) % 10000 == 0:
                print(f"  Processed {i + 1}/{n_samples} samples")
        
        # Determine output path
        if output_name is None:
            suffix_parts = []
            
            if self.with_stats:
                per_class_suffix = "_perclass" if self.per_class_stats else ""
                suffix_parts.append(f"stats_{self.ablation_variant}{per_class_suffix}")
            
            if self.with_comparison:
                balance_suffix = "_balanced" if self.balance_comparison else ""
                pos_suffix = f"_{self.comparison_position}"
                suffix_parts.append(f"comparison_n{self.comparison_n}{balance_suffix}{pos_suffix}")
            
            suffix = "_with_" + "_".join(suffix_parts) if suffix_parts else ""
            task_suffix = f"_task{self.task}" if self.task else ""
            mode_suffix = f"_mode_{self.mode}" if self.mode != "zeroshot" else ""
            output_name = f"{self.dataset_name}_{split}_seed{self.seed}{task_suffix}{suffix}{mode_suffix}.json"
        output_path = os.path.join(self.output_dir, output_name)
        
        # Save
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(samples, f, ensure_ascii=False, indent=2)
        
        print(f"Saved {len(samples)} samples to {output_path}")
        
        # Save statistics separately if enabled (task-agnostic)
        if self.with_stats and stats is not None:
            per_class_suffix = "_perclass" if self.per_class_stats else ""
            task_suffix = f"_task{self.task}" if self.task else ""
            stats_path = os.path.join(
                self.output_dir,
                f"{self.dataset_name}_train_stats_{self.ablation_variant}{per_class_suffix}{task_suffix}.json"
            )
            stats.save(stats_path)
        
        return output_path
    
    def build_all_splits(
        self,
        limit: Optional[int] = None,
    ) -> List[str]:
        """
        Build corpus for all splits (train, valid, test).
        
        Args:
            limit: Maximum number of samples per split
            
        Returns:
            List of paths to saved corpus files
        """
        paths = []
        for split in ["train", "valid", "test"]:
            path = self.build_corpus(split, limit=limit)
            paths.append(path)
        return paths
    


# ==================== CLI ====================

def parse_args():
    parser = argparse.ArgumentParser(
        description="Build Alpaca-style corpus from tabular datasets"
    )
    parser.add_argument(
        "--dataset", "-d",
        type=str,
        required=True,
        help=f"Dataset name. Available: {list(DATASET_REGISTRY.keys())}"
    )
    parser.add_argument(
        "--split", "-s",
        type=str,
        default="train",
        choices=["train", "valid", "test", "all"],
        help="Which split to build (default: train)"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for split files (default: 42)"
    )
    parser.add_argument(
        "--limit", "-l",
        type=int,
        default=None,
        help="Maximum number of samples per split (default: all)"
    )
    parser.add_argument(
        "--output-dir", "-o",
        type=str,
        default=None,
        help="Output directory (default: dataset_dir/corpus; timestamp subfolder is created)"
    )
    parser.add_argument(
        "--output-name",
        type=str,
        default=None,
        help="Output filename (only for single split)"
    )
    parser.add_argument(
        "--task",
        type=str,
        default=None,
        choices=["regression", "classification"],
        help=(
            "Prediction task type for datasets that support multiple tasks "
            "(e.g., California housing). "
            "If omitted, the dataset-specific default is used."
        ),
    )
    
    # Statistics options
    parser.add_argument(
        "--with-stats",
        action="store_true",
        help="Include dataset statistics in prompts (computed from train set only)"
    )
    parser.add_argument(
        "--inline-feature-stats",
        action="store_true",
        help="Inline per-feature stats in the sample input"
    )
    parser.add_argument(
        "--inline-header-block",
        action="store_true",
        help="Add header + target distribution before inline features"
    )
    parser.add_argument(
        "--no-stats-header-block",
        action="store_true",
        help="Disable the stats header/target distribution block"
    )
    parser.add_argument(
        "--no-stats-block",
        action="store_true",
        help="Do not include the full stats block (useful with inline stats)"
    )
    parser.add_argument(
        "--stats-top-k",
        type=str,
        default="5",
        help="Number of top categories to show for categorical features (default: 5)"
    )
    parser.add_argument(
        "--stats-position",
        type=str,
        default="prefix",
        choices=["prefix", "suffix"],
        help="Where to add statistics in input (default: prefix)"
    )
    parser.add_argument(
        "--ablation-variant",
        type=str,
        default="full",
        choices=list(ABLATION_VARIANTS.keys()),
        help=(
            "Ablation variant for numerical stats. "
            "Options: mean_only (only mean), mean_std (mean+std), "
            "no_minmax (mean+std+percentiles), full (all stats). "
            "(default: full)"
        )
    )
    parser.add_argument(
        "--per-class-stats",
        action="store_true",
        help=(
            "Compute statistics separately for each target class. "
            "This helps LLM understand how feature distributions differ by class."
        )
    )
    
    # Comparison options
    parser.add_argument(
        "--with-comparison",
        action="store_true",
        help=(
            "Include similar historical cases (RAG-style examples) in prompts. "
            "Examples are retrieved from training set based on similarity."
        )
    )
    parser.add_argument(
        "--comparison-n",
        type=int,
        default=16,
        help="Number of similar examples to retrieve per sample (default: 5)"
    )
    parser.add_argument(
        "--comparison-position",
        type=str,
        default="before_query",
        choices=["before_query", "after_query"],
        help=(
            "Where to add comparison examples relative to sample features. "
            "'before_query': examples appear before sample features. "
            "'after_query': examples appear after sample features. "
            "(default: before_query)"
        )
    )
    parser.add_argument(
        "--balance-comparison",
        action="store_true",
        help=(
            "Retrieve balanced positive/negative samples (50%% each). "
            "Within each class, samples are still ranked by similarity."
        )
    )
    parser.add_argument(
        "--include-similarity",
        action="store_true",
        default=True,
        help="Include cosine similarity scores in comparison examples (default: True)"
    )
    parser.add_argument(
        "--no-similarity",
        action="store_true",
        help="Exclude cosine similarity scores from comparison examples"
    )
    
    # Mode options
    parser.add_argument(
        "--mode",
        type=str,
        default="zeroshot",
        choices=["zeroshot", "reasoning", "finetune", "normal_finetune"],
        help=(
            "Corpus mode: "
            "'zeroshot' for standard prediction prompts, "
            "'reasoning' for using dataset-specific instruction_reasoning (skip suffixes), "
            "'finetune' for using finetune instruction with given prediction label, "
            "'normal_finetune' for using INSTRUCTION_NORMAL_FINETUNE with given prediction label, "
            "(default: zeroshot)"
        )
    )
    parser.add_argument(
        "--undersampling",
        action="store_true",
        help=(
            "Zeroshot/finetune/normal_finetune mode + limit only: build samples in repeated 3:1 order "
            "(negative, negative, negative, positive). For binary classification datasets."
        ),
    )
    # Output options
    parser.add_argument(
        "--no-timestamp",
        nargs="?",
        const="",
        default=None,
        type=str,
        help=(
            "Control output subdirectory. "
            "Omit to use timestamp subdir (default). "
            "Use '--no-timestamp' (no value) or '--no-timestamp \"\"' to save directly under output dir. "
            "Use '--no-timestamp RUN_NAME' to save under output_dir/RUN_NAME."
        ),
    )
    
    return parser.parse_args()


def main():
    args = parse_args()
    
    # Validate dataset
    if args.dataset not in DATASET_REGISTRY:
        print(f"Error: Dataset '{args.dataset}' not found.")
        print(f"Available datasets: {list(DATASET_REGISTRY.keys())}")
        return
    
    # Handle similarity flag (--no-similarity overrides default)
    include_similarity = not args.no_similarity
    
    if args.undersampling and args.mode not in {"zeroshot", "finetune", "normal_finetune"}:
        raise ValueError(
            "--undersampling is only supported when --mode zeroshot, --mode finetune, or --mode normal_finetune."
        )
    if args.undersampling and (args.limit is None or args.limit <= 0):
        raise ValueError("--undersampling requires a positive --limit.")
    
    # Build corpus
    run_command = shlex.join(sys.argv)
    builder = CorpusBuilder(
        dataset_name=args.dataset,
        seed=args.seed,
        output_dir=args.output_dir,
        run_command=run_command,
        with_stats=args.with_stats,
        inline_feature_stats=args.inline_feature_stats,
        inline_header_block=args.inline_header_block,
        include_stats_header_block=not args.no_stats_header_block,
        include_stats_block=not args.no_stats_block,
        stats_top_k=args.stats_top_k,
        stats_position=args.stats_position,
        ablation_variant=args.ablation_variant,
        per_class_stats=args.per_class_stats,
        task=args.task,
        # Comparison options
        with_comparison=args.with_comparison,
        comparison_n=args.comparison_n,
        comparison_position=args.comparison_position,
        balance_comparison=args.balance_comparison,
        include_similarity=include_similarity,
        # Mode options
        mode=args.mode,
        undersampling=args.undersampling,
        # Output options
        no_timestamp=args.no_timestamp,
    )
    
    if args.with_stats or args.inline_feature_stats:
        if str(args.stats_top_k).lower() == "all":
            top_k_label = "all categories"
        else:
            top_k_label = f"top-{args.stats_top_k} categories"
        print(f"Statistics enabled: {top_k_label}, position={args.stats_position}")
        if args.inline_feature_stats:
            print(f"Inline feature stats: enabled (stats block: {not args.no_stats_block})")
        print(f"Ablation variant: {args.ablation_variant} -> {ABLATION_VARIANTS[args.ablation_variant]}")
        print(f"Per-class statistics: {args.per_class_stats}")
    
    if args.with_comparison:
        print(f"Comparison enabled: {args.comparison_n} examples, position={args.comparison_position}")
        print(f"Balance comparison: {args.balance_comparison}")
        print(f"Include similarity scores: {include_similarity}")
    
    if args.mode != "zeroshot":
        mode_descriptions = {
            "reasoning": "dataset-specific instruction_reasoning for finetuned model inference",
            "finetune": "finetune instruction with given prediction label",
            "normal_finetune": "INSTRUCTION_NORMAL_FINETUNE with given prediction label",
        }
        print(f"Mode: {args.mode} ({mode_descriptions.get(args.mode, '')})")
    if args.undersampling:
        print("Undersampling enabled: pattern is negative, negative, negative, positive (3:1)")
    
    if args.split == "all":
        paths = builder.build_all_splits(limit=args.limit)
        print(f"\nBuilt {len(paths)} corpus files:")
        for p in paths:
            print(f"  - {p}")
    else:
        path = builder.build_corpus(
            split=args.split,
            limit=args.limit,
            output_name=args.output_name,
        )
        print(f"\nCorpus saved to: {path}")


if __name__ == "__main__":
    main()
