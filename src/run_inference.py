#!/usr/bin/env python
"""
Usage:
    # Using pre-built corpus JSON (recommended)
    python run_inference.py --model /path/to/model --dataset adult \
        --corpus data/adult/corpus/adult_test_seed42.json

    # Using default corpus path
    python run_inference.py --model /path/to/model --dataset adult --split test --seed 42
    
    # With options
    python run_inference.py --model /path/to/model --dataset adult \
        --corpus /path/to/corpus.json --batch-size 64 --limit 1000
    
"""

import os
import sys
import json
import argparse
import logging
from datetime import datetime
import traceback
from pathlib import Path

from vllm import SamplingParams
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from vllm_predictor import VLLMBatchPredictor
from utils.common import DualWriter

# Set up console logging first (file handler added later in main())
LOG_FORMAT = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
logging.basicConfig(
    level=logging.INFO,
    format=LOG_FORMAT,
    force=True
)
logger = logging.getLogger(__name__)


def setup_file_logging(log_file_path: str):
    """Add file handler to root logger so all loggers write to file."""
    root_logger = logging.getLogger()
    file_handler = logging.FileHandler(log_file_path, mode='a')
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(logging.Formatter(LOG_FORMAT))
    root_logger.addHandler(file_handler)
    return file_handler


# ==================== Dataset Registry ====================

DATASET_REGISTRY = {}

# Project/base paths for resolving default corpus directories
_BASE_DIR = Path(__file__).resolve().parents[1]
_DATA_DIR = _BASE_DIR / "data"


def register_dataset(name: str, dataset_class: type, default_corpus_dir: str):
    """Register a dataset with its class and default corpus directory."""
    DATASET_REGISTRY[name] = {
        "class": dataset_class,
        "corpus_dir": default_corpus_dir,
    }


def _load_datasets():
    """Load and register all known datasets."""
    try:
        from data.adult.adult_dataset import AdultDataset
        register_dataset(
            "adult",
            AdultDataset,
            str(_DATA_DIR / "adult" / "corpus"),
        )
    except ImportError as e:
        logger.warning(f"Failed to import AdultDataset: {e}")
        print(traceback.format_exc())  # print full stack trace 

    # California housing dataset (regression + classification via task)
    try:
        from data.california.california_dataset import CaliforniaDataset
        register_dataset(
            "california",
            CaliforniaDataset,
            str(_DATA_DIR / "california" / "corpus"),
        )
    except ImportError as e:
        logger.warning(f"Failed to import CaliforniaDataset: {e}")
        print(traceback.format_exc())

    # Diamonds dataset
    try:
        from data.diamonds.diamonds_dataset import DiamondsDataset
        register_dataset(
            "diamonds",
            DiamondsDataset,
            str(_DATA_DIR / "diamonds" / "corpus"),
        )
    except ImportError as e:
        logger.warning(f"Failed to import DiamondsDataset: {e}")
        print(traceback.format_exc())

    # Home Credit dataset
    try:
        from data.home_credit.home_credit_dataset import HomeCreditDataset
        register_dataset(
            "home_credit",
            HomeCreditDataset,
            str(_DATA_DIR / "home_credit" / "corpus"),
        )
    except ImportError as e:
        logger.warning(f"Failed to import HomeCreditDataset: {e}")
        print(traceback.format_exc())

    # OKCupid STEM dataset
    try:
        from data.okcupid_stem.okcupid_stem_dataset import OkCupidStemDataset
        register_dataset(
            "okcupid_stem",
            OkCupidStemDataset,
            str(_DATA_DIR / "okcupid_stem" / "corpus"),
        )
    except ImportError as e:
        logger.warning(f"Failed to import OkCupidStemDataset: {e}")
        print(traceback.format_exc())

    # Diabetes 130-US dataset
    try:
        from data.diabetes_130US.diabetes_130US_dataset import Diabetes130USDataset
        register_dataset(
            "diabetes_130US",
            Diabetes130USDataset,
            str(_DATA_DIR / "diabetes_130US" / "corpus"),
        )
    except ImportError as e:
        logger.warning(f"Failed to import Diabetes130USDataset: {e}")
        print(traceback.format_exc())


_load_datasets()


def get_dataset_class(name: str):
    """Get dataset class by name."""
    if name not in DATASET_REGISTRY:
        raise ValueError(f"Unknown dataset: {name}. Available: {list(DATASET_REGISTRY.keys())}")
    return DATASET_REGISTRY[name]["class"]


def get_default_corpus_path(
    dataset_name: str,
    split: str,
    seed: int,
    task: str | None = None,
) -> str:
    """
    Get default corpus path for a dataset.
    
    If `task` is provided (e.g., "regression" or "classification"),
    the filename will include a `_task{task}` suffix, consistent with
    `corpus_builder.py`.
    """
    if dataset_name not in DATASET_REGISTRY:
        raise ValueError(f"Unknown dataset: {dataset_name}")
    corpus_dir = DATASET_REGISTRY[dataset_name]["corpus_dir"]
    task_suffix = f"_task{task}" if task else ""
    filename = f"{dataset_name}_{split}_seed{seed}{task_suffix}.json"
    return os.path.join(corpus_dir, filename)


# ==================== Main ====================

def parse_args():
    parser = argparse.ArgumentParser(description='Run vLLM batch prediction on tabular datasets')
    
    # Model arguments
    parser.add_argument('--model', '-m', type=str, required=True, help='Model name or path')
    parser.add_argument('--tensor_parallel_size', type=int, default=1, help='Number of GPUs')
    parser.add_argument('--max_model_len', type=int, default=32768, help='Max context length')
    parser.add_argument('--gpu_memory_utilization', type=float, default=0.9, help='GPU memory utilization')
    parser.add_argument('--dtype', type=str, default='auto', help='Model dtype')
    
    # Data arguments
    parser.add_argument('--dataset', '-d', type=str, required=True, help='Dataset name')
    parser.add_argument('--split', '-s', type=str, default='test', choices=['train', 'valid', 'test'], help='Dataset split')
    parser.add_argument('--seed', type=int, default=42, help='Random seed for splits')
    parser.add_argument('--corpus', '-c', type=str, default=None, help='Path to corpus JSON file')
    parser.add_argument('--limit', '-l', type=int, default=None, help='Limit number of samples')
    parser.add_argument('--api_base', type=str, default=None, help='API base URL')
    parser.add_argument(
        '--task',
        type=str,
        default=None,
        choices=['regression', 'classification'],
        help='Task type for datasets that support multiple tasks (e.g., California)',
    )
    # Inference arguments
    parser.add_argument('--batch_size', type=int, default=256, help='Batch size')
    parser.add_argument('--max_tokens', type=int, default=16384, help='Max tokens to generate')
    parser.add_argument('--temperature', type=float, default=0.6, help='Sampling temperature')
    parser.add_argument('--top_p', type=float, default=0.95, help='Top-p (nucleus) sampling')
    
    # Output arguments
    parser.add_argument('--output_dir', '-o', type=str, default=str(_BASE_DIR / "results"), help='Output directory')
    parser.add_argument('--run_group', type=str, default='debug', help='Run group name')
    
    return parser.parse_args()


def main():
    args = parse_args()
    
    # Validate dataset
    if args.dataset not in DATASET_REGISTRY:
        logger.error(f"Unknown dataset: {args.dataset}. Available: {list(DATASET_REGISTRY.keys())}")
        return
    
    # Get dataset class
    dataset_class = get_dataset_class(args.dataset)
    logger.info(f"Using dataset: {args.dataset}, class: {dataset_class.__name__}")
    
    # Determine corpus path
    if args.corpus:
        corpus_path = args.corpus
    else:
        corpus_path = get_default_corpus_path(
            dataset_name=args.dataset,
            split=args.split,
            seed=args.seed,
            task=args.task,
        )
    
    if not os.path.exists(corpus_path):
        logger.error(f"Corpus file not found: {corpus_path}")
        logger.info("Please build corpus first using corpus_builder.py")
        return
    
    logger.info(f"Using corpus: {corpus_path}")
    
    model_name = os.path.basename(os.path.normpath(args.model))
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    args.output_dir = os.path.join(args.output_dir, args.dataset, args.run_group,  model_name, timestamp)
    os.makedirs(args.output_dir, exist_ok=True)
    logger.info(f"Output directory: {args.output_dir}")
    
    # Set up dual logging: both terminal and file
    log_file_path = os.path.join(args.output_dir, "log.txt")
    
    # Add file handler to all loggers (run_inference + vllm_predictor)
    setup_file_logging(log_file_path)
    
    # Redirect print() to both terminal and file
    sys.stdout = DualWriter(log_file_path)
    
    print(args)
    print(f'save path: {args.output_dir}')
    
    # Initialize predictor
    logger.info("Initializing vLLM predictor...")
    predictor = VLLMBatchPredictor(
        model_name=args.model,
        max_model_len=args.max_model_len,
        gpu_memory_utilization=args.gpu_memory_utilization,
        tensor_parallel_size=args.tensor_parallel_size,
        dtype=args.dtype,
        dataset_class=dataset_class,
        api_base=args.api_base,
    )
        
    # Set up sampling parameters
    sampling_params = SamplingParams(
        max_tokens=args.max_tokens,
        temperature=args.temperature,
        top_p=args.top_p,
    )
        
    # Run prediction
    logger.info("Starting batch prediction...")
    results = predictor.predict_corpus(
        corpus_path=corpus_path,
        batch_size=args.batch_size,
        sampling_params=sampling_params,
        limit=args.limit,
        task=args.task,
    )
    logger.info(f"Args: {json.dumps(vars(args), indent=2, default=str)}")
        
    logger.info(f"Prediction completed. Processed {len(results['results'])} samples.")
        
    # Log metrics
    logger.info("\n" + "=" * 60)
    logger.info("EVALUATION RESULTS")
    logger.info("=" * 60)
    if results.get('accuracy') is not None:
        logger.info("Classification metrics:")
        logger.info(f"  Accuracy:  {results['accuracy']:.4f}")
        if results.get('balanced_accuracy') is not None:
            logger.info(f"  Balanced-Accuracy: {results['balanced_accuracy']:.4f}")
        if results.get('precision') is not None:
            logger.info(f"  Precision: {results['precision']:.4f}")
        if results.get('recall') is not None:
            logger.info(f"  Recall:    {results['recall']:.4f}")
        if results.get('f1') is not None:
            logger.info(f"  F1:        {results['f1']:.4f}")
        if results.get('precision_macro') is not None:
            logger.info(f"  Precision-macro: {results['precision_macro']:.4f}")
        if results.get('recall_macro') is not None:
            logger.info(f"  Recall-macro:    {results['recall_macro']:.4f}")
        if results.get('f1_macro') is not None:
            logger.info(f"  F1-macro:        {results['f1_macro']:.4f}")
        logger.info(f"  MCC:       {results['mcc']:.4f}")
        if all(results.get(k) is not None for k in ('tn', 'fp', 'fn', 'tp')):
            logger.info(
                f"  Confusion Matrix: TN={results['tn']}, FP={results['fp']}, "
                f"FN={results['fn']}, TP={results['tp']}"
            )
        if results.get('confusion_matrix') is not None:
            if results.get('confusion_matrix_labels') is not None:
                logger.info(f"  Confusion Matrix Labels: {results['confusion_matrix_labels']}")
            logger.info(
                f"  Confusion Matrix (rows=true, cols=pred): {results['confusion_matrix']}"
            )
    if results.get('rmse') is not None:
        logger.info("Regression metrics:")
        logger.info(f"  RMSE: {results['rmse']:.4f}")
        logger.info(f"  MSE:  {results['mse']:.4f}")
        logger.info(f"  MAE:  {results['mae']:.4f}")
        logger.info(f"  R2:   {results['r2']:.4f}")
    if results.get('missing_count'):
        logger.info(f"Invalid predictions: {results['missing_count']}")
    logger.info("=" * 60)
        
    # Determine data source name for file naming
    data_source = os.path.basename(corpus_path).replace(".json", "")
        
    # Save results
    output_path = os.path.join(args.output_dir, f"predictions_{data_source}.json")
    predictor.save_results(results['results'], output_path)
        
    # Save metrics
    metrics = {
        "model": args.model,
        "dataset": args.dataset,
        "corpus": corpus_path,
        "split": args.split,
        "task": args.task,
        "seed": args.seed,
        "n_samples": len(results['results']),
        # Classification metrics (may be None for regression)
        "accuracy": results.get('accuracy'),
        "balanced_accuracy": results.get('balanced_accuracy'),
        "precision": results.get('precision'),
        "recall": results.get('recall'),
        "f1": results.get('f1'),
        "precision_macro": results.get('precision_macro'),
        "recall_macro": results.get('recall_macro'),
        "f1_macro": results.get('f1_macro'),
        "mcc": results.get('mcc'),
        "tn": results.get('tn'),
        "fp": results.get('fp'),
        "fn": results.get('fn'),
        "tp": results.get('tp'),
        "confusion_matrix": results.get('confusion_matrix'),
        "confusion_matrix_labels": results.get('confusion_matrix_labels'),
        # Regression metrics (may be None for classification)
        "rmse": results.get('rmse'),
        "mse": results.get('mse'),
        "mae": results.get('mae'),
        "r2": results.get('r2'),
        "missing_count": results.get('missing_count'),
        "timestamp": datetime.now().strftime("%Y%m%d_%H%M%S"),
    }
    metrics_path = os.path.join(args.output_dir, f"metrics_{data_source}.json")
    with open(metrics_path, 'w') as f:
        json.dump(metrics, f, indent=2)
    logger.info(f"Metrics saved to: {metrics_path}")
        
    # Save config
    config = vars(args)
    config['corpus_path'] = corpus_path
    config_path = os.path.join(args.output_dir, f"config_{data_source}.json")
    with open(config_path, 'w') as f:
        json.dump(config, f, indent=2)
    logger.info(f"Config saved to: {config_path}")

    logger.info("Batch prediction completed successfully!")


if __name__ == "__main__":
    main()
