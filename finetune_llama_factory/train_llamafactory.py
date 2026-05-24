#!/usr/bin/env python3
"""
Train with LLaMA-Factory CLI and optionally merge LoRA adapter.
"""

from __future__ import annotations

import argparse
import shlex
import subprocess
from datetime import datetime
from pathlib import Path


def _run(cmd: list[str]) -> None:
    print("[CMD]", " ".join(shlex.quote(c) for c in cmd), flush=True)
    subprocess.run(cmd, check=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="Train and optionally merge with LLaMA-Factory.")
    parser.add_argument(
        "--profile",
        choices=[
            "gemma-3-4b-it",
            "gemma-3-1b-it",
            "gemma-3-270m-it",
            "gemma-3-12b-it",
            "gemma-3-27b-it",
            "qwen3-4b",
            "qwen3-1.7b",
            "qwen3-0.6b",
            "qwen3-8b",
            "qwen3-14b",
            "llama-3.1-8b-instruct",
        ],
        default=None,
        help="Use a predefined training profile.",
    )
    parser.add_argument("--model_name_or_path", required=False, help="Base model path or HF repo")
    parser.add_argument("--dataset", required=True, help="Dataset name in dataset_info.json")
    parser.add_argument(
        "--dataset_dir",
        default=None,
        help="Directory that contains dataset_info.json (default: script_dir/dataset_info)",
    )
    parser.add_argument("--output_dir", required=False, help="Directory to save training outputs")
    parser.add_argument("--template", default="default", help="Prompt template name")
    parser.add_argument("--finetuning_type", default="lora", choices=["lora", "full"], help="Finetune type")
    parser.add_argument("--lora_rank", type=int, default=None)
    parser.add_argument("--lora_alpha", type=int, default=None)
    parser.add_argument("--lora_target", type=str, default=None, help="Comma-separated target modules")
    parser.add_argument("--extra_train_args", type=str, default="", help="Extra args passed to train CLI")

    parser.add_argument("--merge", action="store_true", help="Merge adapter after training")
    parser.add_argument("--merged_output_dir", type=str, default=None, help="Path to save merged model")
    parser.add_argument("--export_size", type=int, default=None, help="Shard size (GB) for export")
    parser.add_argument("--export_legacy_format", action="store_true", help="Export in legacy format")
    parser.add_argument("--extra_export_args", type=str, default="", help="Extra args passed to export CLI")
    parser.add_argument("--cutoff_len", type=int, default=None, help="Max token length for samples")
    parser.add_argument(
        "--num_train_epochs",
        type=float,
        default=None,
        help="Training epochs (profile mode default: 1.0)",
    )
    parser.add_argument(
        "--output_root",
        type=str,
        default=None,
        help="Root directory for profile-based outputs (default: script_dir/saves)",
    )

    args = parser.parse_args()

    if args.profile:
        if args.cutoff_len is None:
            raise ValueError("--cutoff_len is required when using --profile")
        num_train_epochs = args.num_train_epochs if args.num_train_epochs is not None else 1.0
        profile_cfg = {
            "gemma-3-12b-it": {
                "model": "google/gemma-3-12b-it",
                "template": "gemma3",
                "output_name": "Gemma-3-12B-Instruct",
                "per_device_train_batch_size": "4",
                "extra": [
                    "--freeze_vision_tower", "True",
                    "--freeze_multi_modal_projector", "True",
                    "--image_max_pixels", "589824",
                    "--image_min_pixels", "1024",
                    "--video_max_pixels", "65536",
                    "--video_min_pixels", "256",
                ],
            },
            "gemma-3-4b-it": {
                "model": "google/gemma-3-4b-it",
                "template": "gemma3",
                "output_name": "Gemma-3-4B-Instruct",
                "per_device_train_batch_size": "4",
                "extra": [
                    "--freeze_vision_tower", "True",
                    "--freeze_multi_modal_projector", "True",
                    "--image_max_pixels", "589824",
                    "--image_min_pixels", "1024",
                    "--video_max_pixels", "65536",
                    "--video_min_pixels", "256",
                ],
            },
            "gemma-3-1b-it": {
                "model": "google/gemma-3-1b-it",
                "template": "gemma2",
                "output_name": "Gemma-3-1B-Instruct",
                "per_device_train_batch_size": "4",
                "extra": [],
            },
            "gemma-3-270m-it": {
                "model": "google/gemma-3-270m-it",
                "template": "gemma2",
                "output_name": "Gemma-3-270M-Instruct",
                "per_device_train_batch_size": "4",
                "extra": [],
            },
            "gemma-3-27b-it": {
                "model": "google/gemma-3-27b-it",
                "template": "gemma3",
                "output_name": "Gemma-3-27B-Instruct",
                "per_device_train_batch_size": "2",
                "extra": [
                    "--freeze_vision_tower", "True",
                    "--freeze_multi_modal_projector", "True",
                    "--image_max_pixels", "589824",
                    "--image_min_pixels", "1024",
                    "--video_max_pixels", "65536",
                    "--video_min_pixels", "256",
                ],
            },
            "qwen3-8b": {
                "model": "Qwen/Qwen3-8B",
                "template": "qwen3",
                "output_name": "Qwen3-8B-Thinking",
                "per_device_train_batch_size": "4",
                "extra": [],
            },
            "qwen3-4b": {
                "model": "Qwen/Qwen3-4B",
                "template": "qwen3",
                "output_name": "Qwen3-4B-Thinking",
                "per_device_train_batch_size": "4",
                "extra": [],
            },
            "qwen3-1.7b": {
                "model": "Qwen/Qwen3-1.7B",
                "template": "qwen3",
                "output_name": "Qwen3-1.7B-Thinking",
                "per_device_train_batch_size": "4",
                "extra": [],
            },
            "qwen3-0.6b": {
                "model": "Qwen/Qwen3-0.6B",
                "template": "qwen3",
                "output_name": "Qwen3-0.6B-Thinking",
                "per_device_train_batch_size": "4",
                "extra": [],
            },
            "qwen3-14b": {
                "model": "Qwen/Qwen3-14B",
                "template": "qwen3",
                "output_name": "Qwen3-14B-Thinking",
                "per_device_train_batch_size": "4",
                "extra": [],
            },
            "llama-3.1-8b-instruct": {
                "model": "meta-llama/Meta-Llama-3.1-8B-Instruct",
                "template": "llama3",
                "output_name": "Llama-3.1-8B-Instruct",
                "per_device_train_batch_size": "4",
                "extra": [],
            },
        }[args.profile]

        script_dir = Path(__file__).resolve().parent
        dataset_dir = (Path(args.dataset_dir) if args.dataset_dir else script_dir / "dataset_info").resolve()
        output_root = (Path(args.output_root) if args.output_root else script_dir / "saves").resolve()

        timestamp = datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
        output_dir = (
            output_root
            / profile_cfg["output_name"]
            / "lora"
            / f"train_{timestamp}"
        ).resolve()
        output_dir.mkdir(parents=True, exist_ok=True)

        train_cmd = [
            "llamafactory-cli",
            "train",
            "--stage", "sft",
            "--do_train", "True",
            "--model_name_or_path", profile_cfg["model"],
            "--preprocessing_num_workers", "16",
            "--finetuning_type", "lora",
            "--template", profile_cfg["template"],
            "--flash_attn", "auto",
            "--dataset_dir", str(dataset_dir),
            "--dataset", args.dataset,
            "--cutoff_len", str(args.cutoff_len),
            "--learning_rate", "1e-04",
            "--num_train_epochs", str(num_train_epochs),
            "--max_samples", "100000",
            "--per_device_train_batch_size", profile_cfg["per_device_train_batch_size"],
            "--gradient_accumulation_steps", "8",
            "--lr_scheduler_type", "cosine",
            "--max_grad_norm", "1.0",
            "--logging_steps", "5",
            "--save_steps", "100",
            "--warmup_steps", "0",
            "--packing", "False",
            "--enable_thinking", "True",
            "--report_to", "none",
            "--output_dir", str(output_dir),
            "--bf16", "True",
            "--plot_loss", "True",
            "--trust_remote_code", "True",
            "--ddp_timeout", "180000000",
            "--include_num_input_tokens_seen", "True",
            "--optim", "adamw_torch",
            "--lora_rank", "8",
            "--lora_alpha", "16",
            "--lora_dropout", "0",
            "--lora_target", "all",
        ] + profile_cfg["extra"]

        _run(train_cmd)
        return

    if not args.model_name_or_path:
        raise ValueError("--model_name_or_path is required when --profile is not set")
    if not args.output_dir:
        raise ValueError("--output_dir is required when --profile is not set")

    dataset_dir = Path(args.dataset_dir).resolve()
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    train_cmd = [
        "llamafactory-cli",
        "train",
        "--model_name_or_path",
        args.model_name_or_path,
        "--dataset_dir",
        str(dataset_dir),
        "--dataset",
        args.dataset,
        "--template",
        args.template,
        "--finetuning_type",
        args.finetuning_type,
        "--output_dir",
        str(output_dir),
    ]

    if args.num_train_epochs is not None:
        train_cmd += ["--num_train_epochs", str(args.num_train_epochs)]

    if args.finetuning_type == "lora":
        if args.lora_rank is not None:
            train_cmd += ["--lora_rank", str(args.lora_rank)]
        if args.lora_alpha is not None:
            train_cmd += ["--lora_alpha", str(args.lora_alpha)]
        if args.lora_target:
            train_cmd += ["--lora_target", args.lora_target]

    if args.extra_train_args:
        train_cmd += shlex.split(args.extra_train_args)

    _run(train_cmd)

    if not args.merge:
        return

    merged_output_dir = args.merged_output_dir
    if not merged_output_dir:
        raise ValueError("--merged_output_dir is required when --merge is set")

    export_cmd = [
        "llamafactory-cli",
        "export",
        "--model_name_or_path",
        args.model_name_or_path,
        "--adapter_name_or_path",
        str(output_dir),
        "--template",
        args.template,
        "--export_dir",
        str(Path(merged_output_dir).resolve()),
    ]
    if args.export_size is not None:
        export_cmd += ["--export_size", str(args.export_size)]
    if args.export_legacy_format:
        export_cmd += ["--export_legacy_format"]
    if args.extra_export_args:
        export_cmd += shlex.split(args.extra_export_args)

    _run(export_cmd)


if __name__ == "__main__":
    main()
