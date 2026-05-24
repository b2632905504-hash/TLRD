import json
import os
import re
from typing import List, Dict, Any, Optional, Literal

import torch
from torch.utils.data import Dataset as TorchDataset, DataLoader
from vllm import LLM, SamplingParams
import numpy as np
import pandas as pd
from tqdm import tqdm
import logging
from transformers import AutoTokenizer
from openai import OpenAI
from sklearn.metrics import (
    matthews_corrcoef, precision_score, recall_score, f1_score,
    classification_report, accuracy_score, confusion_matrix, balanced_accuracy_score,
    mean_squared_error, mean_absolute_error, r2_score,
)

# Set up logging
logger = logging.getLogger(__name__)


class QueryDataset(TorchDataset):
    """PyTorch Dataset for handling queries/prompts"""
    
    def __init__(self, queries: List[str], indices: Optional[List[int]] = None):
        self.queries = queries
        self.indices = indices if indices is not None else list(range(len(queries)))
    
    def __len__(self):
        return len(self.queries)
    
    def __getitem__(self, idx):
        return {
            'query': self.queries[idx],
            'index': self.indices[idx]
        }


def load_corpus_json(corpus_path: str) -> List[Dict[str, Any]]:
    with open(corpus_path, "r", encoding="utf-8") as f:
        return json.load(f)


class VLLMBatchPredictor:
    """
    Batch predictor using vLLM for tabular dataset classification.
    """
    
    def __init__(
        self,
        model_name: str,
        max_model_len: int = 4096,
        gpu_memory_utilization: float = 0.9,
        tensor_parallel_size: int = 1,
        trust_remote_code: bool = True,
        dtype: str = "auto",
        dataset_class: Optional[type] = None,
        api_base: Optional[str] = None,
    ):
        """
        Initialize the vLLM batch predictor.
        
        Args:
            model_name: Name or path of the model to use
            max_model_len: Maximum sequence length for the model
            gpu_memory_utilization: GPU memory utilization ratio
            tensor_parallel_size: Number of GPUs for tensor parallelism
            trust_remote_code: Whether to trust remote code in the model
            dtype: Model dtype (auto, float16, bfloat16)
            dataset_class: Dataset class with parse_output method for output parsing
            api_base: Optional URL of external vLLM server (e.g., "http://localhost:8000/v1")
                      If provided, will use the external server instead of loading model locally
        """
        self.model_name = model_name
        self.max_model_len = max_model_len
        self.gpu_memory_utilization = gpu_memory_utilization
        self.tensor_parallel_size = tensor_parallel_size
        self.trust_remote_code = trust_remote_code
        self.dtype = dtype
        self.dataset_class = dataset_class
        self.api_base = api_base
        
        # Initialize tokenizer and model (or API client)
        self.tokenizer = None
        self.llm = None
        self.client = None  # OpenAI client for external vLLM server
        self._initialize_model()
    
    def _initialize_model(self):
        """Initialize the vLLM model and tokenizer, or connect to external server."""
        try:
            # Always initialize tokenizer for chat template
            logger.info(f"Initializing tokenizer for model: {self.model_name}")
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.model_name,
                trust_remote_code=self.trust_remote_code
            )
            logger.info("Tokenizer initialized successfully")
            
            if self.api_base:
                # Use external vLLM server via OpenAI-compatible API
                logger.info(f"Using external vLLM server at: {self.api_base}")
                self.client = OpenAI(
                    base_url=self.api_base,
                    api_key="EMPTY",  # vLLM doesn't require API key by default
                )
                logger.info("Connected to external vLLM server successfully")
            else:
                # Load model locally
                logger.info(f"Initializing local vLLM model: {self.model_name}")
                
                # Check CUDA availability
                if torch.cuda.is_available():
                    logger.info(f"CUDA available. GPU count: {torch.cuda.device_count()}")
                    self.tensor_parallel_size = min(self.tensor_parallel_size, torch.cuda.device_count())
                else:
                    logger.warning("CUDA not available, using CPU")
                    self.tensor_parallel_size = 1
                
                self.llm = LLM(
                    model=self.model_name,
                    max_model_len=self.max_model_len,
                    gpu_memory_utilization=self.gpu_memory_utilization,
                    tensor_parallel_size=self.tensor_parallel_size,
                    trust_remote_code=self.trust_remote_code,
                    dtype='auto' if self.dtype == 'fp8' else self.dtype,
                )
                logger.info("Local vLLM model initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize vLLM model: {e}")
            raise
    
    def _apply_chat_template(self, system_prompt: str, content: str) -> str:
        """
        Apply chat template to content.
        
        Args:
            content: The message content (instruction + input)
            
        Returns:
            Formatted prompt string
        """
        if self.tokenizer is None:
            raise RuntimeError("Tokenizer not initialized")
        
        messages = [{"role": "system", "content": system_prompt}, {"role": "user", "content": content}]
        
        prompt = self.tokenizer.apply_chat_template(
            messages, 
            tokenize=False, 
            add_generation_prompt=True
        )
        return prompt
    
    def predict_dataset(
        self,
        dataset: pd.DataFrame,
        prompt_column: str = "prompt",
        batch_size: int = 32,
        sampling_params: Optional[SamplingParams] = None,
        max_tokens: int = 5120,
        temperature: float = 0.0,
        task: Optional[Literal["regression", "classification"]] = None,
    ) -> Dict[str, Any]:
        """
        Make predictions on a pandas DataFrame.
        
        Args:
            dataset: pandas DataFrame with prompt and target columns
            prompt_column: Column name containing the prompts (already formatted)
            batch_size: Batch size for processing
            sampling_params: Sampling parameters for generation (used for local vLLM)
            max_tokens: Maximum tokens for generation (used for API mode)
            temperature: Temperature for generation (used for API mode)
            
        Returns:
            Dictionary containing prediction results and metrics
        """
        if self.llm is None and self.client is None:
            raise RuntimeError("Model not initialized")
        
        # Default sampling params for local mode
        if sampling_params is None:
            sampling_params = SamplingParams(
                max_tokens=max_tokens,
                temperature=temperature,
                top_p=1.0,
            )
        
        # Extract prompts and indices
        prompts = dataset[prompt_column].tolist()
        indices = dataset['index'].tolist() if 'index' in dataset.columns else list(range(len(prompts)))
        indices = [int(i) for i in indices]

        def _split_response(raw_response: str) -> tuple[Optional[str], str]:
            text = (raw_response or "").strip()
            if 'gpt-oss' in self.model_name:
                if 'assistantfinal' not in text:
                    return None, text
                parts = text.split('assistantfinal', 1)
                return parts[0].strip(), parts[1].strip() if len(parts) > 1 else text
            if 'qwen' in self.model_name:
                sep = '</think>' if '</think>' in text else '<\/think>'
                if sep not in text:
                    return None, text
                parts = text.split(sep, 1)
                return parts[0].strip(), parts[1].strip() if len(parts) > 1 else text
            return None, text

        def _parse_label(text: Optional[str]) -> Optional[str]:
            if text is None:
                return None
            try:
                return self.dataset_class.parse_output(text, task=task)
            except TypeError:
                return self.dataset_class.parse_output(text)

        def _run_inference(
            current_prompts: List[str],
            current_indices: List[int],
            current_sampling_params: SamplingParams,
            current_max_tokens: int,
            current_temperature: float,
            desc: str,
        ) -> List[Dict[str, Any]]:
            query_dataset = QueryDataset(current_prompts, current_indices)
            dataloader = DataLoader(query_dataset, batch_size=batch_size, shuffle=False)
            batch_results: List[Dict[str, Any]] = []

            for batch in tqdm(dataloader, desc=desc):
                batch_prompts = list(batch['query'])
                batch_indices = [int(i) for i in batch['index']]

                if self.api_base and self.client:
                    try:
                        response = self.client.completions.create(
                            model=self.model_name,
                            prompt=batch_prompts,
                            max_tokens=current_max_tokens,
                            temperature=current_temperature,
                            top_p=getattr(current_sampling_params, "top_p", 1.0),
                        )
                        processed = min(len(response.choices), len(batch_prompts))
                        for j in range(processed):
                            raw_response = (response.choices[j].text or "").strip()
                            response_think, response_without_think = _split_response(raw_response)
                            label = _parse_label(response_without_think)
                            if label is None:
                                logger.warning(f"Unparsed prediction at index {batch_indices[j]}")
                            batch_results.append({
                                "input": batch_prompts[j],
                                "index": batch_indices[j],
                                "label": label,
                                "response_think": response_think,
                                "response_without_think": response_without_think,
                            })
                        for j in range(processed, len(batch_prompts)):
                            batch_results.append({
                                "input": batch_prompts[j],
                                "index": batch_indices[j],
                                "label": None,
                                "response_think": None,
                                "response_without_think": None,
                            })
                    except Exception as e:
                        logger.error(f"Batch API call failed: {e}")
                        for j in range(len(batch_prompts)):
                            batch_results.append({
                                "input": batch_prompts[j],
                                "index": batch_indices[j],
                                "label": None,
                                "response_think": None,
                                "response_without_think": None,
                            })
                else:
                    outputs = self.llm.generate(batch_prompts, current_sampling_params)
                    for j, output in enumerate(outputs):
                        raw_response = output.outputs[0].text.strip() if output.outputs else ""
                        response_think, response_without_think = _split_response(raw_response)
                        label = _parse_label(response_without_think)
                        if label is None:
                            logger.warning(f"Unparsed prediction at index {batch_indices[j]}")
                        batch_results.append({
                            "input": batch_prompts[j],
                            "index": batch_indices[j],
                            "label": label,
                            "response_think": response_think,
                            "response_without_think": response_without_think,
                        })
            return batch_results

        # First-pass inference
        results = _run_inference(
            current_prompts=prompts,
            current_indices=indices,
            current_sampling_params=sampling_params,
            current_max_tokens=max_tokens,
            current_temperature=temperature,
            desc="Processing batches",
        )

        # Auto-rerun for unparsed predictions (fixed policy)
        # Stop when missing ratio <= 5%, or after 10 extra rounds,
        # or when improvement stagnates for 2 consecutive checks.
        max_missing_ratio = 0.05
        max_rerun_rounds = 10
        stagnation_patience = 2
        total_count = len(indices)
        min_missing_drop = max(1, int(total_count * 0.001))
        index_to_prompt = {idx: prompt for idx, prompt in zip(indices, prompts)}
        results_by_index = {int(item["index"]): item for item in results}
        last_missing_count: Optional[int] = None
        stagnation_rounds = 0

        for rerun_round in range(1, max_rerun_rounds + 1):
            missing_indices = [
                idx for idx in indices
                if idx not in results_by_index or results_by_index[idx].get("label") is None
            ]
            missing_count = len(missing_indices)
            missing_ratio = (missing_count / total_count) if total_count > 0 else 0.0
            logger.info(
                f"Unparsed check after round {rerun_round - 1}: "
                f"{missing_count}/{total_count} ({missing_ratio:.2%})"
            )
            if missing_count == 0 or missing_ratio <= max_missing_ratio:
                break

            if last_missing_count is not None:
                improved = last_missing_count - missing_count
                if improved < min_missing_drop:
                    stagnation_rounds += 1
                    logger.info(
                        f"Rerun improvement too small ({improved} < {min_missing_drop}); "
                        f"stagnation {stagnation_rounds}/{stagnation_patience}"
                    )
                else:
                    stagnation_rounds = 0
                if stagnation_rounds >= stagnation_patience:
                    logger.info("Stopping reruns early due to stalled improvement")
                    break

            logger.info(f"Rerun round {rerun_round} for {missing_count} unparsed samples")
            rerun_prompts = [index_to_prompt[idx] for idx in missing_indices]
            rerun_temperature = max(float(getattr(sampling_params, "temperature", temperature)), 0.2)
            rerun_sampling_params = SamplingParams(
                max_tokens=int(getattr(sampling_params, "max_tokens", max_tokens)),
                temperature=rerun_temperature,
                top_p=float(getattr(sampling_params, "top_p", 1.0)),
            )
            rerun_results = _run_inference(
                current_prompts=rerun_prompts,
                current_indices=missing_indices,
                current_sampling_params=rerun_sampling_params,
                current_max_tokens=int(getattr(rerun_sampling_params, "max_tokens", max_tokens)),
                current_temperature=float(getattr(rerun_sampling_params, "temperature", rerun_temperature)),
                desc=f"Rerun round {rerun_round}",
            )
            for item in rerun_results:
                results_by_index[int(item["index"])] = item
            last_missing_count = missing_count

        results = [
            results_by_index.get(
                idx,
                {
                    "input": index_to_prompt[idx],
                    "index": idx,
                    "label": None,
                    "response_think": None,
                    "response_without_think": None,
                },
            )
            for idx in indices
        ]
        
        # Create results DataFrame
        results_df = pd.DataFrame(results)
        logger.info(f"Label distribution:\n{results_df['label'].value_counts()}")
        
        # Compute metrics if target column exists
        if 'target' in dataset.columns:
            results_df = results_df.merge(
                dataset[['index', 'target']],
                on='index',
                how='left'
            )

            # Infer task if not provided
            inferred_task = task
            if inferred_task is None:
                tgt = results_df['target']
                if pd.api.types.is_numeric_dtype(tgt):
                    uniq = pd.unique(tgt.dropna())
                    uniq_set = set(uniq)
                    # Treat small-cardinality integer-coded targets as classification
                    if uniq_set.issubset({0, 1}):
                        inferred_task = "classification"
                    elif len(uniq) <= 20 and np.all(np.isclose(uniq, np.round(uniq))):
                        inferred_task = "classification"
                    else:
                        inferred_task = "regression"
                else:
                    inferred_task = "classification"

            # Filter valid predictions
            valid_mask = results_df['label'].notna()
            missing_count = int((~valid_mask).sum())

            if missing_count > 0:
                logger.warning(f"Skipping {missing_count} predictions due to missing/None labels")
                logger.warning(results_df[~valid_mask])

            if valid_mask.sum() == 0:
                logger.info("No valid predictions")
                return {
                    'results': results_df,
                    'accuracy': None,
                    'precision': None,
                    'recall': None,
                    'f1': None,
                    'mcc': None,
                    'rmse': None,
                    'mse': None,
                    'mae': None,
                    'r2': None,
                    'classification_report': None,
                    'missing_count': missing_count
                }

            valid_results = results_df[valid_mask].copy()

            if inferred_task == "regression":
                logger.info("Regression task detected")
                preds = []
                for i, lbl in enumerate(valid_results['label']):
                    try:
                        preds.append(self.dataset_class.output_to_target(lbl, task="regression"))
                        if preds[-1] is None:
                            logger.warning(f"Skipping prediction {i} due to None label", lbl)
                    except TypeError:
                        preds.append(self.dataset_class.output_to_target(lbl))
                        if preds[-1] is None:
                            logger.warning(f"Skipping prediction {i} due to None label", lbl)
                valid_results['prediction'] = preds

                # Filter out rows where output_to_target returned NaN/None
                pred_valid_mask = pd.Series(valid_results['prediction']).notna()
                if (~pred_valid_mask).sum() > 0:
                    logger.warning(f"Skipping {(~pred_valid_mask).sum()} predictions that could not be converted to valid labels")
                    missing_count += int((~pred_valid_mask).sum())
                valid_results = valid_results[pred_valid_mask.values].copy()

                if len(valid_results) == 0:
                    logger.info("No valid predictions after label conversion")
                    return {
                        'results': results_df,
                        'accuracy': None,
                        'precision': None,
                        'recall': None,
                        'f1': None,
                        'mcc': None,
                        'rmse': None,
                        'mse': None,
                        'mae': None,
                        'r2': None,
                        'classification_report': None,
                        'missing_count': missing_count
                    }

                y_true = valid_results['target'].astype(float)
                y_pred = pd.Series(valid_results['prediction']).astype(float)

                rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
                mse = float(mean_squared_error(y_true, y_pred))
                mae = float(mean_absolute_error(y_true, y_pred))
                r2 = float(r2_score(y_true, y_pred))

                return {
                    'results': results_df,
                    'rmse': rmse,
                    'mse': mse,
                    'mae': mae,
                    'r2': r2,
                    'accuracy': None,
                    'precision': None,
                    'recall': None,
                    'f1': None,
                    'mcc': None,
                    'classification_report': None,
                    'missing_count': missing_count
                }

            # Classification path (binary or multiclass)
            pred_ids = []
            for lbl in valid_results['label']:
                try:
                    pred_ids.append(self.dataset_class.output_to_target(lbl, task="classification"))
                except TypeError:
                    pred_ids.append(self.dataset_class.output_to_target(lbl))
            valid_results['prediction_id'] = pred_ids

            # Filter out rows where output_to_target returned NaN/None
            pred_valid_mask = pd.Series(valid_results['prediction_id']).notna()
            if (~pred_valid_mask).sum() > 0:
                logger.warning(f"Skipping {(~pred_valid_mask).sum()} predictions that could not be converted to valid labels")
                missing_count += int((~pred_valid_mask).sum())
            valid_results = valid_results[pred_valid_mask.values].copy()
            
            if len(valid_results) == 0:
                logger.info("No valid predictions after label conversion")
                return {
                    'results': results_df,
                    'accuracy': None,
                    'precision': None,
                    'recall': None,
                    'f1': None,
                    'mcc': None,
                    'rmse': None,
                    'mse': None,
                    'mae': None,
                    'r2': None,
                    'classification_report': None,
                    'missing_count': missing_count
                }

            valid_results['prediction_id'] = valid_results['prediction_id'].astype(int)
            y_true = valid_results['target'].astype(int)
            y_pred = valid_results['prediction_id']

            n_classes = len(pd.unique(y_true))

            accuracy = accuracy_score(y_true, y_pred)
            balanced_acc = balanced_accuracy_score(y_true, y_pred)
            if n_classes <= 2:
                precision = precision_score(y_true, y_pred, zero_division=0)
                recall = recall_score(y_true, y_pred, zero_division=0)
                f1 = f1_score(y_true, y_pred, zero_division=0)
            else:
                precision = None
                recall = None
                f1 = None
            mcc = matthews_corrcoef(y_true, y_pred)
            precision_macro = precision_score(y_true, y_pred, average='macro', zero_division=0)
            recall_macro = recall_score(y_true, y_pred, average='macro', zero_division=0)
            f1_macro = f1_score(y_true, y_pred, average='macro', zero_division=0)

            # Always compute full confusion matrix; keep binary [0,1] convention for compatibility.
            if n_classes <= 2:
                cm_label_ids = [0, 1]
            else:
                cm_label_ids = sorted(
                    pd.unique(pd.concat([y_true, y_pred], ignore_index=True)).astype(int).tolist()
                )

            cm = confusion_matrix(y_true, y_pred, labels=cm_label_ids)
            confusion_matrix_values = cm.astype(int).tolist()

            def _format_cm_label(label_id: int) -> str:
                if self.dataset_class is not None and hasattr(self.dataset_class, "target_to_output"):
                    try:
                        return str(self.dataset_class.target_to_output(label_id))
                    except Exception:
                        pass
                return str(label_id)

            confusion_matrix_labels = [_format_cm_label(label_id) for label_id in cm_label_ids]

            tn = fp = fn = tp = None
            if n_classes <= 2 and cm.shape == (2, 2):
                tn, fp, fn, tp = cm.ravel()

            report = classification_report(y_true, y_pred)
            logger.info(f"Classification Report:\n{report}")
            if n_classes <= 2:
                logger.info(
                    f"Accuracy: {accuracy:.4f}, Balanced-Accuracy: {balanced_acc:.4f}, "
                    f"Precision: {precision:.4f}, Recall: {recall:.4f}, F1: {f1:.4f}, "
                    f"Precision-macro: {precision_macro:.4f}, Recall-macro: {recall_macro:.4f}, "
                    f"F1-macro: {f1_macro:.4f}, MCC: {mcc:.4f}"
                )
            else:
                logger.info(
                    f"Accuracy: {accuracy:.4f}, Balanced-Accuracy: {balanced_acc:.4f}, "
                    f"Precision-macro: {precision_macro:.4f}, Recall-macro: {recall_macro:.4f}, "
                    f"F1-macro: {f1_macro:.4f}, MCC: {mcc:.4f}"
                )

            return {
                'results': results_df,
                'accuracy': accuracy,
                'balanced_accuracy': balanced_acc,
                'precision': precision,
                'recall': recall,
                'f1': f1,
                'precision_macro': precision_macro,
                'recall_macro': recall_macro,
                'f1_macro': f1_macro,
                'mcc': mcc,
                'tn': int(tn) if tn is not None else None,
                'fp': int(fp) if fp is not None else None,
                'fn': int(fn) if fn is not None else None,
                'tp': int(tp) if tp is not None else None,
                'confusion_matrix': confusion_matrix_values,
                'confusion_matrix_labels': confusion_matrix_labels,
                'rmse': None,
                'mse': None,
                'mae': None,
                'r2': None,
                'classification_report': report,
                'missing_count': missing_count
            }
        else:
            logger.error("Target column not found in dataset")
        return {
            'results': results_df,
            'accuracy': None,
            'precision': None,
            'recall': None,
            'f1': None,
            'mcc': None,
            'rmse': None,
            'mse': None,
            'mae': None,
            'r2': None,
            'classification_report': None,
            'missing_count': int(results_df['label'].isna().sum())
        }
    
    def predict_corpus(
        self,
        corpus_path: str,
        batch_size: int = 32,
        sampling_params: Optional[SamplingParams] = None,
        limit: Optional[int] = None,
        task: Optional[Literal["regression", "classification"]] = None,
    ) -> Dict[str, Any]:
        """
        Make predictions on an Alpaca-style corpus JSON file.
        
        The corpus JSON contains pre-built prompts with instruction, input, and output.
        Prompts are loaded directly and formatted with chat template.
        
        Args:
            corpus_path: Path to the corpus JSON file
            batch_size: Batch size for processing
            sampling_params: Sampling parameters for generation
            limit: Maximum number of samples to process
            Returns:
            Dictionary containing prediction results and metrics
        """
        # Load corpus
        corpus = load_corpus_json(corpus_path)
        logger.info(f"Loaded {len(corpus)} samples from {corpus_path}")
        
        if limit and limit > 0:
            corpus = corpus[:limit]
            logger.warning(f"Limited to {len(corpus)} samples")
        
        # Build prompts directly from corpus (instruction + input -> apply chat template)
        prompts = []
        targets = []
        for sample in tqdm(corpus, desc="Preparing prompts"):
            system_prompt = sample["instruction"] if "instruction" in sample else self.dataset_class.INSTRUCTION
            content = sample["input"]
            # Apply chat template
            prompt = self._apply_chat_template(system_prompt, content)
            prompts.append(prompt)
            
            # Parse target from output using dataset-specific method if available
            output_text = sample.get("output", "")
            try:
                target_int = self.dataset_class.output_to_target(output_text, task=task)
            except TypeError:
                target_int = self.dataset_class.output_to_target(output_text)

            # Fallback: parse free-form output text first, then map to target id.
            if target_int is None:
                try:
                    target = self.dataset_class.parse_output(output_text, task=task)
                except TypeError:
                    target = self.dataset_class.parse_output(output_text)
                try:
                    target_int = self.dataset_class.output_to_target(target, task=task)
                except TypeError:
                    target_int = self.dataset_class.output_to_target(target)
            targets.append(target_int)
        
        # Create DataFrame
        df = pd.DataFrame({
            'prompt': prompts,
            'target': targets,
            'index': list(range(len(prompts)))
        })
        
        # Use predict_dataset
        return self.predict_dataset(
            dataset=df,
            prompt_column='prompt',
            batch_size=batch_size,
            sampling_params=sampling_params,
            task=task,
        )
    
    def save_results(self, results: pd.DataFrame, output_path: str):
        """
        Save prediction results to files.
        
        Args:
            results: DataFrame of prediction results
            output_path: Path to save the results (JSON format)
        """
        # Handle both DataFrame and dict input
        if isinstance(results, dict):
            results = results if isinstance(results, pd.DataFrame) else pd.DataFrame(results)
        
        # Save as CSV
        csv_path = output_path.replace('.json', '.csv')
        results.to_csv(csv_path, index=False)
        logger.info(f"Results saved to CSV: {csv_path}")
        
        # Save as JSON
        results.to_json(output_path, orient='records', indent=2)
        logger.info(f"Results saved to JSON: {output_path}")
    
    def __del__(self):
        """Cleanup when the object is destroyed."""
        if hasattr(self, 'llm') and self.llm is not None:
            del self.llm
        if hasattr(self, 'tokenizer') and self.tokenizer is not None:
            del self.tokenizer
        if hasattr(self, 'client') and self.client is not None:
            del self.client
