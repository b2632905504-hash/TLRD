#!/usr/bin/env python
"""
Build a finetuning corpus by distilling a teacher model.

Flow:
1) Load existing Alpaca-style corpus JSON (instruction/input/output).
2) Build teacher prompts using the corpus instruction and an input that
   appends the given prediction label at the end of the current sample.
3) Run teacher model via vLLM and save outputs as the finetune corpus output.
4) Save Alpaca-style JSON to data/<dataset>/corpus_finetune.
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import os
import re
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from datetime import datetime

from transformers import AutoTokenizer
from vllm import LLM, SamplingParams

_SCRIPT_DIR = Path(__file__).resolve().parent
if str(_SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPT_DIR))
_PARENT_DIR = _SCRIPT_DIR.parent
if str(_PARENT_DIR) not in sys.path:
    sys.path.insert(0, str(_PARENT_DIR))

from data.dataset import DATASET_REGISTRY  # pylint: disable=import-error
from data.dataset_registry import get_dataset_class  # pylint: disable=import-error


LABEL_LINE_TEMPLATE = "THE GIVEN PREDICTION LABEL is {label}."
LABEL_LINE_RE = re.compile(r"^\s*THE GIVEN PREDICTION LABEL is .*$", re.MULTILINE)
GPT_THOUGHT_START = "<|channel|>analysis<|message|>"
GPT_THOUGHT_END = "<|end|><|start|>assistant<|channel|>final<|message|>"

START_MARKER = "--- Current Sample Features ---"
END_MARKER = "=== Similar Historical Cases ==="

LOG_FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
logging.basicConfig(level=logging.INFO, format=LOG_FORMAT, force=True)
logger = logging.getLogger(__name__)


def _is_corpus_file(path: Path) -> bool:
    if path.suffix != ".json":
        return False
    if "train_stats" in path.name:
        return False
    return True


def _load_corpus(path: Path) -> List[Dict[str, Any]]:
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise ValueError(f"Expected list in {path}, got {type(data)}")
    return data


def _extract_final(text: str) -> str:
    t = text.replace("\r\n", "\n").replace("\r", "\n")
    m = re.search(r"\bassistant\s*final\b", t, flags=re.IGNORECASE)
    if m:
        return t[m.end():].strip()
    m = re.search(r"<\|channel\|>\s*final\b", t, flags=re.IGNORECASE)
    if m:
        return t[m.end():].strip()
    t = re.sub(r"^(analysis|<analysis>)\s*", "", t, flags=re.IGNORECASE).strip()
    return t


def _extract_input_between_markers(text: str) -> Optional[str]:
    if not text:
        return None
    start_idx = text.find(START_MARKER)
    label_match = LABEL_LINE_RE.search(text)
    if not label_match:
        return None
    end_idx = label_match.start()
    if start_idx == -1 or end_idx < start_idx:
        return None
    start_idx += len(START_MARKER)
    return text[start_idx:end_idx].strip()


def _split_think_final(text: str) -> tuple[Optional[str], str]:
    t = text.replace("\r\n", "\n").replace("\r", "\n")

    if GPT_THOUGHT_START in t and GPT_THOUGHT_END in t:
        _, _, rest = t.partition(GPT_THOUGHT_START)
        think, _, final = rest.partition(GPT_THOUGHT_END)
        return think.strip(), final.strip()

    if "assistantfinal" in t:
        think, _, final = t.partition("assistantfinal")
        return think.strip(), final.strip()

    if "</think>" in t or "<\\/think>" in t:
        marker = "</think>" if "</think>" in t else "<\\/think>"
        think, _, final = t.partition(marker)
        return think.strip(), final.strip()

    return None, _extract_final(t)


def _compose_gpt_output(response_think: Optional[str], response_without_think: str) -> str:
    if not response_think:
        return response_without_think
    return f"{GPT_THOUGHT_START}{response_think}{GPT_THOUGHT_END}{response_without_think}"


def _normalize_think_text(text: Optional[str]) -> Optional[str]:
    if text is None:
        return None
    t = text.lstrip()
    if t.lower().startswith("analysis"):
        t = t[len("analysis"):].lstrip()
    return t


def _build_chat_prompt(tokenizer, system_text: str, user_text: str) -> str:
    messages = [
        {"role": "system", "content": system_text},
        {"role": "user", "content": user_text},
    ]
    return tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )


def _append_label_to_input(input_text: str, output_label: str) -> str:
    if input_text and LABEL_LINE_RE.search(input_text):
        return input_text
    label_line = LABEL_LINE_TEMPLATE.format(label=output_label)
    if not input_text:
        return label_line
    if input_text.endswith("\n"):
        return input_text + label_line
    return input_text + "\n\n" + label_line


def _strip_given_label(input_text: str) -> str:
    if not input_text:
        return input_text
    cleaned = LABEL_LINE_RE.sub("", input_text)
    cleaned = re.sub(r"\n{3,}", "\n\n", cleaned).strip()
    return cleaned


def _resolve_reasoning_instruction(
    dataset_class,
    sample: Dict[str, Any],
    task: Optional[str],
    feature_explanations: str,
) -> str:
    if task == "classification":
        if hasattr(dataset_class, "INSTRUCTION_REASONING_CLASSIFICATION"):
            instruction = dataset_class.INSTRUCTION_REASONING_CLASSIFICATION
        elif hasattr(dataset_class, "INSTRUCTION_REASONING"):
            instruction = dataset_class.INSTRUCTION_REASONING
        else:
            instruction = sample.get("instruction", "")
    elif task == "regression":
        if hasattr(dataset_class, "INSTRUCTION_REASONING_REGRESSION"):
            instruction = dataset_class.INSTRUCTION_REASONING_REGRESSION
        elif hasattr(dataset_class, "INSTRUCTION_REASONING"):
            instruction = dataset_class.INSTRUCTION_REASONING
        else:
            instruction = sample.get("instruction", "")
    elif hasattr(dataset_class, "INSTRUCTION_REASONING"):
        instruction = dataset_class.INSTRUCTION_REASONING
    else:
        instruction = sample.get("instruction", "")

    if feature_explanations and "Feature explanations" not in instruction:
        instruction = instruction.rstrip() + "\n\n" + feature_explanations + "\n"
    return instruction


def _resolve_normal_instruction(
    dataset_class,
    sample: Dict[str, Any],
    task: Optional[str],
    feature_explanations: str,
) -> str:
    if task == "classification":
        if hasattr(dataset_class, "INSTRUCTION_CLASSIFICATION"):
            instruction = dataset_class.INSTRUCTION_CLASSIFICATION
        elif hasattr(dataset_class, "INSTRUCTION"):
            instruction = dataset_class.INSTRUCTION
        else:
            instruction = sample.get("instruction", "")
    elif task == "regression":
        if hasattr(dataset_class, "INSTRUCTION_REGRESSION"):
            instruction = dataset_class.INSTRUCTION_REGRESSION
        elif hasattr(dataset_class, "INSTRUCTION"):
            instruction = dataset_class.INSTRUCTION
        else:
            instruction = sample.get("instruction", "")
    elif hasattr(dataset_class, "INSTRUCTION"):
        instruction = dataset_class.INSTRUCTION
    else:
        instruction = sample.get("instruction", "")

    if feature_explanations and "Feature explanations" not in instruction:
        instruction = instruction.rstrip() + "\n\n" + feature_explanations + "\n"
    return instruction


def _normalize_input(sample: Dict[str, Any]) -> str:
    input_text = sample.get("input", "")
    if isinstance(input_text, str):
        extracted = _extract_input_between_markers(input_text)
        if extracted is not None:
            input_text = extracted
    return input_text


def _resolve_finetune_instruction(
    dataset_class,
    sample: Dict[str, Any],
    task: Optional[str],
) -> str:
    if hasattr(dataset_class, "INSTRUCTION_FIENTUNE"):
        return dataset_class.INSTRUCTION_FIENTUNE
    if task == "classification" and hasattr(dataset_class, "FINETUNE_INSTRUCTION_CLASSIFICATION"):
        return dataset_class.FINETUNE_INSTRUCTION_CLASSIFICATION
    if task == "regression" and hasattr(dataset_class, "FINETUNE_INSTRUCTION_REGRESSION"):
        return dataset_class.FINETUNE_INSTRUCTION_REGRESSION
    if hasattr(dataset_class, "FINETUNE_INSTRUCTION"):
        return dataset_class.FINETUNE_INSTRUCTION
    return sample.get("instruction", "")


def _build_teacher_prompts(
    tokenizer,
    dataset_class,
    corpus: List[Dict[str, Any]],
    task: Optional[str],
) -> List[str]:
    prompts = []
    for sample in corpus:
        system_text = sample.get("instruction", "")
        input_text = sample.get("input", "")
        output_label = str(sample.get("output", "")).strip()
        user_text = _append_label_to_input(input_text, output_label)
        prompts.append(_build_chat_prompt(tokenizer, system_text, user_text))
    return prompts


def _build_teacher_messages(
    corpus: List[Dict[str, Any]],
) -> List[List[Dict[str, str]]]:
    messages = []
    for sample in corpus:
        system_text = sample.get("instruction", "")
        input_text = sample.get("input", "")
        output_label = str(sample.get("output", "")).strip()
        user_text = _append_label_to_input(input_text, output_label)
        messages.append(
            [
                {"role": "system", "content": system_text},
                {"role": "user", "content": user_text},
            ]
        )
    return messages


def _build_finetune_corpus_with_outputs(
    dataset_class,
    corpus: List[Dict[str, Any]],
    teacher_outputs: List[str],
    task: Optional[str],
    feature_explanations: str,
    normal: bool = False,
) -> List[Dict[str, str]]:
    finetune = []
    for sample, teacher_out in zip(corpus, teacher_outputs):
        input_text = _normalize_input(sample)
        if normal:
            instruction = _resolve_normal_instruction(
                dataset_class,
                sample,
                task,
                feature_explanations,
            )
        else:
            instruction = _resolve_reasoning_instruction(
                dataset_class,
                sample,
                task,
                feature_explanations,
            )
        finetune.append({
            "instruction": instruction,
            "input": _strip_given_label(input_text),
            "output": teacher_out,
        })
    return finetune


def _parse_output_with_dataset(
    dataset_class,
    raw_text: str,
    task: Optional[str],
) -> Any:
    try:
        return dataset_class.parse_output(raw_text, task=task)
    except TypeError:
        return dataset_class.parse_output(raw_text)


def _collect_retry_candidate_indices(
    dataset_class,
    corpus: List[Dict[str, Any]],
    outputs_final: List[str],
    task: Optional[str],
) -> List[int]:
    if len(corpus) != len(outputs_final):
        raise ValueError(
            f"Mismatched lengths for retry-candidate check: corpus={len(corpus)}, outputs_final={len(outputs_final)}"
        )
    retry_indices: List[int] = []

    def _to_float(value: Any) -> Optional[float]:
        if value is None:
            return None
        if isinstance(value, (int, float)):
            return float(value)
        s = str(value).strip().replace(",", "")
        if not s:
            return None
        try:
            return float(s)
        except ValueError:
            return None

    for idx, (sample, teacher_out_final) in enumerate(zip(corpus, outputs_final)):
        expected_raw = str(sample.get("output", ""))
        expected_label = expected_raw.strip().lower()
        if not expected_label:
            continue
        predicted_parsed = _parse_output_with_dataset(dataset_class, teacher_out_final, task)
        if predicted_parsed is None:
            retry_indices.append(idx)
            continue

        expected_float = _to_float(expected_raw) if task == "regression" else None
        predicted_float = _to_float(predicted_parsed) if task == "regression" else None
        if task == "regression" and expected_float is not None and predicted_float is not None:
            labels_match = math.isclose(expected_float, predicted_float, rel_tol=1e-9, abs_tol=1e-6)
        else:
            labels_match = str(predicted_parsed).strip().lower() == expected_label

        if not labels_match:
            retry_indices.append(idx)
    return retry_indices


def _filter_by_label_consistency(
    dataset_class,
    corpus: List[Dict[str, Any]],
    outputs_text: List[str],
    outputs_final: List[str],
    task: Optional[str],
) -> Tuple[List[Dict[str, Any]], List[str], List[str], Dict[str, int], List[Dict[str, Any]]]:
    if not (len(corpus) == len(outputs_text) == len(outputs_final)):
        raise ValueError(
            f"Mismatched lengths for filtering: corpus={len(corpus)}, "
            f"outputs_text={len(outputs_text)}, outputs_final={len(outputs_final)}"
        )

    kept_corpus: List[Dict[str, Any]] = []
    kept_outputs_text: List[str] = []
    kept_outputs_final: List[str] = []
    filtered_records: List[Dict[str, Any]] = []
    stats = {
        "total": len(corpus),
        "kept": 0,
        "filtered": 0,
        "expected_empty": 0,
        "predicted_parse_fail": 0,
        "label_mismatch": 0,
    }

    def _to_float(value: Any) -> Optional[float]:
        if value is None:
            return None
        if isinstance(value, (int, float)):
            return float(value)
        s = str(value).strip().replace(",", "")
        if not s:
            return None
        try:
            return float(s)
        except ValueError:
            return None

    for sample, teacher_out_text, teacher_out_final in zip(corpus, outputs_text, outputs_final):
        expected_raw = str(sample.get("output", ""))
        predicted_parsed = _parse_output_with_dataset(dataset_class, teacher_out_final, task)
        expected_label = expected_raw.strip().lower()
        if not expected_label:
            stats["expected_empty"] += 1
            stats["filtered"] += 1
            filtered_records.append({
                "reason": "expected_empty",
                "expected_raw_output": expected_raw,
                "predicted_parsed_output": None,
                "teacher_output_final": teacher_out_final,
                "teacher_output_merged": teacher_out_text,
            })
            continue
        if predicted_parsed is None:
            stats["predicted_parse_fail"] += 1
            stats["filtered"] += 1
            filtered_records.append({
                "reason": "predicted_parse_fail",
                "expected_raw_output": expected_raw,
                "predicted_parsed_output": None,
                "teacher_output_final": teacher_out_final,
                "teacher_output_merged": teacher_out_text,
            })
            continue

        expected_float = _to_float(expected_raw) if task == "regression" else None
        predicted_float = _to_float(predicted_parsed) if task == "regression" else None
        if task == "regression" and expected_float is not None and predicted_float is not None:
            labels_match = math.isclose(expected_float, predicted_float, rel_tol=1e-9, abs_tol=1e-6)
            expected_cmp = str(expected_float)
            predicted_cmp = str(predicted_float)
        else:
            expected_cmp = expected_label
            predicted_cmp = str(predicted_parsed).strip().lower()
            labels_match = predicted_cmp == expected_cmp

        if not labels_match:
            stats["label_mismatch"] += 1
            stats["filtered"] += 1
            filtered_records.append({
                "reason": "label_mismatch",
                "expected_raw_output": expected_raw,
                "expected_parsed_output": expected_cmp,
                "predicted_parsed_output": predicted_cmp,
                "teacher_output_final": teacher_out_final,
                "teacher_output_merged": teacher_out_text,
            })
            continue

        kept_corpus.append(sample)
        kept_outputs_text.append(teacher_out_text)
        kept_outputs_final.append(teacher_out_final)
        stats["kept"] += 1

    return kept_corpus, kept_outputs_text, kept_outputs_final, stats, filtered_records


def _resolve_corpus_paths(
    corpus_path: Optional[str],
    corpus_dir: Optional[str],
    recursive: bool,
    dataset_dir: Path,
) -> List[Path]:
    if corpus_path:
        return [Path(corpus_path)]
    base_dir = Path(corpus_dir) if corpus_dir else dataset_dir / "corpus"
    if recursive:
        return [p for p in base_dir.rglob("*.json") if _is_corpus_file(p)]
    return [p for p in base_dir.glob("*.json") if _is_corpus_file(p)]


def _write_args_json(args_path: Path, args_dict: Dict[str, Any]) -> None:
    with args_path.open("w", encoding="utf-8") as f:
        json.dump(args_dict, f, ensure_ascii=False, indent=2)


def _load_openai_checkpoint(checkpoint_path: Path) -> List[str]:
    if not checkpoint_path.exists():
        return []
    outputs: List[str] = []
    with checkpoint_path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
                outputs.append(str(rec.get("output", "")))
            except Exception as exc:  # pylint: disable=broad-except
                logger.warning(
                    "skip invalid checkpoint line: file=%s line=%d err=%s",
                    checkpoint_path,
                    line_no,
                    exc,
                )
    return outputs


def _append_openai_checkpoint(checkpoint_path: Path, start_idx: int, outputs: List[str]) -> None:
    with checkpoint_path.open("a", encoding="utf-8") as f:
        for offset, output in enumerate(outputs):
            rec = {"idx": start_idx + offset, "output": output}
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")


def _setup_file_logging(log_path: Path) -> logging.Handler:
    root_logger = logging.getLogger()
    file_handler = logging.FileHandler(log_path, mode="a", encoding="utf-8")
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(logging.Formatter(LOG_FORMAT))
    root_logger.addHandler(file_handler)
    return file_handler


def _build_args_for_logging(args: argparse.Namespace) -> Dict[str, Any]:
    args_dict = vars(args).copy()

    # Never persist sensitive fields.
    for key in ("openai_api_key", "openai_base_url"):
        args_dict.pop(key, None)

    teacher = args_dict.get("teacher")
    if teacher == "vllm":
        for key in (
            "api",
            "openai_concurrency",
            "openai_timeout",
            "openai_max_retries",
        ):
            args_dict.pop(key, None)
    elif teacher == "openai":
        for key in (
            "dtype",
            "trust_remote_code",
            "tensor_parallel_size",
            "max_model_len",
            "gpu_memory_utilization",
            "batch_size",
        ):
            args_dict.pop(key, None)

    return args_dict


def _get_feature_explanations(dataset_class) -> str:
    try:
        dataset = dataset_class(split="train", seed=42)
        if hasattr(dataset, "_build_feature_explanations"):
            return dataset._build_feature_explanations() or ""
    except Exception:
        return ""
    return ""


def _resolve_effective_task(cli_task: Optional[str], dataset_task_type: str) -> str:
    if cli_task:
        return cli_task
    return "regression" if dataset_task_type == "regression" else "classification"


def main() -> None:
    parser = argparse.ArgumentParser(description="Build finetune corpus via teacher model.")
    parser.add_argument("--dataset", required=True, help=f"Dataset name. Available: {list(DATASET_REGISTRY.keys())}")
    parser.add_argument("--corpus-path", type=str, default=None, help="Input corpus JSON file")
    parser.add_argument("--corpus-dir", type=str, default=None, help="Input corpus directory")
    parser.add_argument("--recursive", action="store_true", help="Search corpus-dir recursively")
    parser.add_argument("--out-dir", type=str, default=None, help="Output directory")
    parser.add_argument("--output-suffix", type=str, default=None, help="Suffix for output filenames")
    parser.add_argument(
        "--no-timestamp",
        nargs="?",
        const="",
        default=None,
        type=str,
        help=(
            "Control output subdirectory. "
            "Omit to use timestamp subdir (default). "
            "Use '--no-timestamp' (no value) or '--no-timestamp \"\"' to save directly under out dir. "
            "Use '--no-timestamp RUN_NAME' to save under out_dir/RUN_NAME."
        ),
    )
    parser.add_argument(
        "--task",
        type=str,
        default=None,
        choices=["regression", "classification"],
        help="Task type for datasets that support multiple tasks",
    )
    parser.add_argument(
        "--normal",
        action="store_true",
        help="Use the dataset's plain instruction in the final output corpus instead of the dataset reasoning instruction.",
    )

    parser.add_argument("--model", default="openai/gpt-oss-20b", help="HF model name or local path (vLLM teacher)")
    parser.add_argument("--api", default=None, help="OpenAI/Azure deployment name (OpenAI teacher)")
    parser.add_argument("--dtype", default="auto", help="vLLM dtype: auto/half/bfloat16/float16 etc.")
    parser.add_argument("--trust-remote-code", action="store_true", help="Pass trust_remote_code to tokenizer/vLLM")
    parser.add_argument("--tensor-parallel-size", type=int, default=1, help="TP size if multi-GPU")
    parser.add_argument("--max-model-len", type=int, default=32768)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.90)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--max-tokens", type=int, default=16384)
    parser.add_argument("--temperature", type=float, default=0.6)
    parser.add_argument("--top-p", type=float, default=0.95)
    parser.add_argument("--teacher", choices=["vllm", "openai"], default="vllm")
    parser.add_argument("--openai-base-url", type=str, default=None)
    parser.add_argument("--openai-api-key", type=str, default=None)
    parser.add_argument("--openai-timeout", type=float, default=None)
    parser.add_argument("--openai-max-retries", type=int, default=2)
    parser.add_argument(
        "--resume-openai",
        action="store_true",
        help="Resume OpenAI generation from checkpoint file if present",
    )
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--end", type=int, default=-1)
    parser.add_argument("--num-samples", type=int, default=None, help="Limit number of samples to process")
    parser.add_argument("--log-n", type=int, default=10)
    parser.add_argument(
        "--skip-save-filtered",
        action="store_true",
        help="Do not save filtered-out records JSON",
    )
    parser.add_argument("--test", action="store_true", help="Only build first 10 samples for quick check")
    args = parser.parse_args()

    dataset_name = args.dataset
    dataset_class = get_dataset_class(dataset_name)
    cfg = DATASET_REGISTRY[dataset_name]
    effective_task = _resolve_effective_task(args.task, cfg.task_type)
    dataset_dir = Path(cfg.csv_path).parent
    feature_explanations = _get_feature_explanations(dataset_class)
    logger.info(
        "task resolved: cli_task=%s, dataset_task_type=%s, effective_task=%s",
        args.task,
        cfg.task_type,
        effective_task,
    )

    corpus_paths = _resolve_corpus_paths(
        args.corpus_path,
        args.corpus_dir,
        args.recursive,
        dataset_dir,
    )
    if not corpus_paths:
        raise FileNotFoundError("No corpus files found.")

    is_qwen_model = "qwen" in args.model.lower()
    tokenizer = None
    llm = None
    sampling_params = None
    if args.teacher == "vllm":
        tokenizer = AutoTokenizer.from_pretrained(
            args.model,
            trust_remote_code=args.trust_remote_code,
            use_fast=True,
        )

        llm = LLM(
            model=args.model,
            dtype=args.dtype,
            trust_remote_code=args.trust_remote_code,
            tensor_parallel_size=args.tensor_parallel_size,
            max_model_len=args.max_model_len,
            gpu_memory_utilization=args.gpu_memory_utilization,
            max_num_seqs=args.batch_size,
        )

        sampling_params = SamplingParams(
            temperature=args.temperature,
            top_p=args.top_p,
            max_tokens=args.max_tokens,
        )

    if args.teacher == "openai" and args.api:
        model_clean = re.sub(r"[\\/-]", "_", args.api)
    else:
        model_clean = re.sub(r"[\\/-]", "_", args.model)
    output_suffix = args.output_suffix or f"finetune_{model_clean}"
    if args.out_dir:
        # If output dir is explicitly provided, default to writing directly into it.
        base_output_dir = Path(args.out_dir)
        use_timestamp_default = False
    elif args.corpus_path:
        # If a single input corpus file is specified, default to writing outputs
        # in the same directory as that corpus, without creating a timestamp subdir.
        base_output_dir = Path(args.corpus_path).resolve().parent
        use_timestamp_default = False
    else:
        base_output_dir = dataset_dir / "corpus_finetune"
        use_timestamp_default = True

    # Output directory selection:
    # - args.no_timestamp is None:
    #   - save directly for explicit --out-dir or --corpus-path inputs
    #   - use timestamp subdir for dataset-level default outputs
    # - args.no_timestamp is "": save directly to base dir
    # - args.no_timestamp is "some_str": save to base_dir/some_str
    if args.no_timestamp is None:
        if use_timestamp_default:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_dir = base_output_dir / f"{timestamp}_{model_clean}"
        else:
            output_dir = base_output_dir
    elif args.no_timestamp == "":
        output_dir = base_output_dir
    else:
        output_dir = base_output_dir / args.no_timestamp
    output_dir.mkdir(parents=True, exist_ok=True)
    log_path = output_dir / "log.txt"
    file_handler = _setup_file_logging(log_path)
    logger.info("file logging enabled -> %s", log_path)

    # Save args to output_dir
    args_dict = _build_args_for_logging(args)
    args_dict["task"] = effective_task
    args_dict["label_consistency_stats"] = {
        "per_file": [],
        "overall": {
            "total": 0,
            "kept": 0,
            "filtered": 0,
            "expected_empty": 0,
            "predicted_parse_fail": 0,
            "label_mismatch": 0,
        },
    }
    args_path = output_dir / "args.json"
    _write_args_json(args_path, args_dict)
    logger.info("saved args -> %s", args_path)

    for path in sorted(corpus_paths):
        corpus = _load_corpus(path)

        start = max(0, args.start)
        end = len(corpus) if args.end == -1 else min(len(corpus), args.end)
        if args.num_samples is not None:
            if args.num_samples <= 0:
                raise ValueError("--num-samples must be > 0")
            end = min(end, start + args.num_samples)
        if args.test:
            start = 0
            end = min(len(corpus), 10)
        if start >= end:
            raise ValueError(f"Invalid slice: start={start}, end={end}, total={len(corpus)}")

        corpus_slice = corpus[start:end]
        logger.info(
            "processing corpus file=%s total=%d slice=[%d:%d] slice_size=%d",
            path,
            len(corpus),
            start,
            end,
            len(corpus_slice),
        )
        outputs_text: List[str] = []
        outputs_final: List[str] = []
        logs: List[Dict[str, Any]] = []
        messages_list: Optional[List[List[Dict[str, str]]]] = None
        prompts: Optional[List[str]] = None
        generate_chat_outputs_fn = None

        if args.teacher == "openai":
            from src.openai_teacher import generate_chat_outputs as generate_chat_outputs_fn  # pylint: disable=import-error
            messages_list = _build_teacher_messages(corpus_slice)
            checkpoint_path = output_dir / f"{path.stem}_{output_suffix}_openai_outputs.jsonl"
            if checkpoint_path.exists() and not args.resume_openai:
                checkpoint_path.unlink()
                logger.info("removed stale openai checkpoint -> %s", checkpoint_path)

            outputs: List[str] = _load_openai_checkpoint(checkpoint_path) if args.resume_openai else []
            if len(outputs) > len(messages_list):
                logger.warning(
                    "checkpoint longer than target slice (%d > %d); truncating",
                    len(outputs),
                    len(messages_list),
                )
                outputs = outputs[: len(messages_list)]

            if outputs:
                logger.info(
                    "resuming openai generation from checkpoint: %d/%d done -> %s",
                    len(outputs),
                    len(messages_list),
                    checkpoint_path,
                )

            pending_start = len(outputs)
            if pending_start < len(messages_list):
                pending_messages = messages_list[pending_start:]
                pending_outputs = generate_chat_outputs_fn(
                    messages_list=pending_messages,
                    model=args.api or "",
                    base_url=args.openai_base_url,
                    api_key=args.openai_api_key,
                    temperature=args.temperature,
                    top_p=args.top_p,
                    max_completion_tokens=args.max_tokens,
                    timeout=args.openai_timeout,
                    max_retries=args.openai_max_retries,
                    fail_on_error=False,
                )
                outputs.extend(pending_outputs)
                _append_openai_checkpoint(checkpoint_path, pending_start, pending_outputs)
                empty_outputs = sum(1 for x in pending_outputs if not x.strip())
                logger.info(
                    "generated openai pending: %d/%d (new=%d, empty_outputs=%d)",
                    len(outputs),
                    len(messages_list),
                    len(pending_outputs),
                    empty_outputs,
                )

            if len(outputs) < len(messages_list):
                missing = len(messages_list) - len(outputs)
                logger.warning("openai outputs missing %d items; filling with empty strings", missing)
                outputs.extend([""] * missing)

            outputs_text = outputs[: len(messages_list)]
            outputs_final = outputs_text
            if args.log_n > 0:
                for k in range(min(args.log_n, len(messages_list))):
                    logs.append({
                        "idx": k,
                        "teacher_prompt": messages_list[k],
                        "raw_output": outputs_text[k],
                        "response_think": None,
                        "response_without_think": outputs_text[k],
                        "merged_output": outputs_text[k],
                    })
            logger.info("generated %d/%d", len(outputs_text), len(messages_list))
        else:
            prompts = _build_teacher_prompts(
                tokenizer,
                dataset_class,
                corpus_slice,
                effective_task,
            )
            total_batches = (len(prompts) + args.batch_size - 1) // args.batch_size

            for batch_idx, i in enumerate(range(0, len(prompts), args.batch_size), start=1):
                batch_prompts = prompts[i:i + args.batch_size]
                batch_out = llm.generate(batch_prompts, sampling_params)

                for j, out in enumerate(batch_out):
                    raw_out = out.outputs[0].text
                    response_think, response_without_think = _split_think_final(raw_out)
                    response_think = _normalize_think_text(response_think)
                    if is_qwen_model:
                        # Qwen outputs may include <think> blocks; keep only the content after </think>.
                        merged_out = response_without_think
                    else:
                        merged_out = _compose_gpt_output(response_think, response_without_think)
                    outputs_text.append(merged_out)
                    outputs_final.append(response_without_think)

                    k = i + j
                    if args.log_n > 0 and k < args.log_n:
                        logs.append({
                            "idx": k,
                            "teacher_prompt": batch_prompts[j],
                            "raw_output": raw_out,
                            "response_think": response_think,
                            "response_without_think": response_without_think,
                            "merged_output": merged_out,
                        })

                done = min(i + args.batch_size, len(prompts))
                logger.info(
                    "generated batch %d/%d (%d/%d)",
                    batch_idx,
                    total_batches,
                    done,
                    len(prompts),
                )

        # Auto-rerun for outputs that would be filtered out.
        # Stop when retry-candidate ratio <= 5%, or after 10 rounds,
        # or when improvement stalls for 2 consecutive checks.
        rerun_max_rounds = 10
        rerun_target_ratio = 0.05
        rerun_stagnation_patience = 2
        rerun_min_drop = max(1, int(len(corpus_slice) * 0.001))
        rerun_last_count: Optional[int] = None
        rerun_stagnation_rounds = 0
        rerun_rounds_executed = 0
        rerun_final_retry_candidates = 0
        rerun_temperature = max(float(args.temperature), 0.2)
        rerun_sampling_params = None
        if args.teacher == "vllm":
            rerun_sampling_params = SamplingParams(
                temperature=rerun_temperature,
                top_p=args.top_p,
                max_tokens=args.max_tokens,
            )

        for rerun_round in range(1, rerun_max_rounds + 1):
            retry_indices = _collect_retry_candidate_indices(
                dataset_class=dataset_class,
                corpus=corpus_slice,
                outputs_final=outputs_final,
                task=effective_task,
            )
            retry_count = len(retry_indices)
            retry_ratio = (retry_count / len(corpus_slice)) if corpus_slice else 0.0
            rerun_final_retry_candidates = retry_count
            logger.info(
                "filter-candidate check after round %d: %d/%d (%.2f%%)",
                rerun_round - 1,
                retry_count,
                len(corpus_slice),
                retry_ratio * 100.0,
            )
            if retry_count == 0 or retry_ratio <= rerun_target_ratio:
                break

            if rerun_last_count is not None:
                improved = rerun_last_count - retry_count
                if improved < rerun_min_drop:
                    rerun_stagnation_rounds += 1
                    logger.info(
                        "filter-candidate improvement too small (%d < %d); stagnation %d/%d",
                        improved,
                        rerun_min_drop,
                        rerun_stagnation_rounds,
                        rerun_stagnation_patience,
                    )
                else:
                    rerun_stagnation_rounds = 0
                if rerun_stagnation_rounds >= rerun_stagnation_patience:
                    logger.info("stopping rerun early due to stalled improvement")
                    break

            logger.info("rerun round %d for %d filter-candidate samples", rerun_round, retry_count)
            rerun_rounds_executed = rerun_round

            if args.teacher == "openai":
                if messages_list is None or generate_chat_outputs_fn is None:
                    raise RuntimeError("OpenAI teacher state not initialized for rerun.")
                pending_messages = [messages_list[idx] for idx in retry_indices]
                pending_outputs = generate_chat_outputs_fn(
                    messages_list=pending_messages,
                    model=args.api or "",
                    base_url=args.openai_base_url,
                    api_key=args.openai_api_key,
                    temperature=rerun_temperature,
                    top_p=args.top_p,
                    max_completion_tokens=args.max_tokens,
                    timeout=args.openai_timeout,
                    max_retries=args.openai_max_retries,
                    fail_on_error=False,
                )
                if len(pending_outputs) < len(pending_messages):
                    missing = len(pending_messages) - len(pending_outputs)
                    logger.warning("openai rerun outputs missing %d items; filling with empty strings", missing)
                    pending_outputs.extend([""] * missing)
                for local_idx, sample_idx in enumerate(retry_indices):
                    rerun_text = pending_outputs[local_idx]
                    outputs_text[sample_idx] = rerun_text
                    outputs_final[sample_idx] = rerun_text
            else:
                if prompts is None or llm is None or rerun_sampling_params is None:
                    raise RuntimeError("vLLM teacher state not initialized for rerun.")
                total_rerun_batches = (len(retry_indices) + args.batch_size - 1) // args.batch_size
                for batch_idx, i in enumerate(range(0, len(retry_indices), args.batch_size), start=1):
                    batch_sample_indices = retry_indices[i:i + args.batch_size]
                    batch_prompts = [prompts[idx] for idx in batch_sample_indices]
                    batch_out = llm.generate(batch_prompts, rerun_sampling_params)
                    for j, out in enumerate(batch_out):
                        raw_out = out.outputs[0].text if out.outputs else ""
                        response_think, response_without_think = _split_think_final(raw_out)
                        response_think = _normalize_think_text(response_think)
                        if is_qwen_model:
                            merged_out = response_without_think
                        else:
                            merged_out = _compose_gpt_output(response_think, response_without_think)
                        sample_idx = batch_sample_indices[j]
                        outputs_text[sample_idx] = merged_out
                        outputs_final[sample_idx] = response_without_think
                    done = min(i + args.batch_size, len(retry_indices))
                    logger.info(
                        "rerun round %d batch %d/%d (%d/%d)",
                        rerun_round,
                        batch_idx,
                        total_rerun_batches,
                        done,
                        len(retry_indices),
                    )

            rerun_last_count = retry_count

        logger.info(
            "rerun summary: rounds=%d, final_filter_candidates=%d/%d (%.2f%%)",
            rerun_rounds_executed,
            rerun_final_retry_candidates,
            len(corpus_slice),
            (rerun_final_retry_candidates / len(corpus_slice) * 100.0) if corpus_slice else 0.0,
        )

        (
            filtered_corpus,
            filtered_outputs_text,
            filtered_outputs_final,
            filter_stats,
            filtered_records,
        ) = _filter_by_label_consistency(
            dataset_class=dataset_class,
            corpus=corpus_slice,
            outputs_text=outputs_text,
            outputs_final=outputs_final,
            task=effective_task,
        )
        logger.info(
            "label-consistency filter: total=%d, kept=%d, filtered=%d, expected_empty=%d, "
            "predicted_parse_fail=%d, label_mismatch=%d",
            filter_stats["total"],
            filter_stats["kept"],
            filter_stats["filtered"],
            filter_stats["expected_empty"],
            filter_stats["predicted_parse_fail"],
            filter_stats["label_mismatch"],
        )
        per_file_stats = {
            "file": str(path),
            "total": filter_stats["total"],
            "kept": filter_stats["kept"],
            "filtered": filter_stats["filtered"],
            "expected_empty": filter_stats["expected_empty"],
            "predicted_parse_fail": filter_stats["predicted_parse_fail"],
            "label_mismatch": filter_stats["label_mismatch"],
            "filtered_count": len(filtered_records),
            "rerun_rounds": rerun_rounds_executed,
            "rerun_final_filter_candidates": rerun_final_retry_candidates,
        }
        if not args.skip_save_filtered:
            per_file_stats["filtered_file"] = str(output_dir / f"{path.stem}_{output_suffix}_filtered.json")
        args_dict["label_consistency_stats"]["per_file"].append(per_file_stats)
        overall_stats = args_dict["label_consistency_stats"]["overall"]
        for key in overall_stats:
            overall_stats[key] += filter_stats[key]
        _write_args_json(args_path, args_dict)
        logger.info("updated args with label_consistency_stats -> %s", args_path)

        if not args.skip_save_filtered:
            filtered_path = output_dir / f"{path.stem}_{output_suffix}_filtered.json"
            with filtered_path.open("w", encoding="utf-8") as f:
                json.dump(filtered_records, f, ensure_ascii=False, indent=2)
            logger.info("saved filtered records %d -> %s", len(filtered_records), filtered_path)

        finetune_final = _build_finetune_corpus_with_outputs(
            dataset_class,
            filtered_corpus,
            filtered_outputs_final,
            effective_task,
            feature_explanations,
            normal=args.normal,
        )

        out_name_final = f"{path.stem}_{output_suffix}_final.json"
        out_path_final = output_dir / out_name_final
        with out_path_final.open("w", encoding="utf-8") as f:
            json.dump(finetune_final, f, ensure_ascii=False, indent=2)
        logger.info("saved %d records -> %s", len(finetune_final), out_path_final)

        if args.log_n > 0:
            log_path = output_dir / f"{path.stem}_{output_suffix}_logs_{args.log_n}.json"
            with log_path.open("w", encoding="utf-8") as f:
                json.dump(logs, f, ensure_ascii=False, indent=2)
            logger.info("saved logs -> %s", log_path)

    logging.getLogger().removeHandler(file_handler)
    file_handler.close()


if __name__ == "__main__":
    main()
