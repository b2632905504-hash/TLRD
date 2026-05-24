#!/usr/bin/env python3
"""
One-click pipeline:
1) finetune via train_llamafactory.py
2) merge LoRA adapter
3) run inference (optional, multi-GPU with repeats)

Usage:
  python train_merge_infer.py --config train_merge_infer.example.json
"""

from __future__ import annotations

import argparse
import json
import os
import re
import shlex
import subprocess
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Any


PROFILE_CFG: dict[str, dict[str, str]] = {
    "gemma-3-4b-it": {
        "model": "google/gemma-3-4b-it",
        "template": "gemma3",
        "output_name": "Gemma-3-4B-Instruct",
    },
    "gemma-3-1b-it": {
        "model": "google/gemma-3-1b-it",
        "template": "gemma2",
        "output_name": "Gemma-3-1B-Instruct",
    },
    "gemma-3-270m-it": {
        "model": "google/gemma-3-270m-it",
        "template": "gemma2",
        "output_name": "Gemma-3-270M-Instruct",
    },
    "gemma-3-12b-it": {
        "model": "google/gemma-3-12b-it",
        "template": "gemma3",
        "output_name": "Gemma-3-12B-Instruct",
    },
    "gemma-3-27b-it": {
        "model": "google/gemma-3-27b-it",
        "template": "gemma3",
        "output_name": "Gemma-3-27B-Instruct",
    },
    "qwen3-8b": {
        "model": "Qwen/Qwen3-8B",
        "template": "qwen3",
        "output_name": "Qwen3-8B-Thinking",
    },
    "qwen3-4b": {
        "model": "Qwen/Qwen3-4B",
        "template": "qwen3",
        "output_name": "Qwen3-4B-Thinking",
    },
    "qwen3-1.7b": {
        "model": "Qwen/Qwen3-1.7B",
        "template": "qwen3",
        "output_name": "Qwen3-1.7B-Thinking",
    },
    "qwen3-0.6b": {
        "model": "Qwen/Qwen3-0.6B",
        "template": "qwen3",
        "output_name": "Qwen3-0.6B-Thinking",
    },
    "qwen3-14b": {
        "model": "Qwen/Qwen3-14B",
        "template": "qwen3",
        "output_name": "Qwen3-14B-Thinking",
    },
    "llama-3.1-8b-instruct": {
        "model": "meta-llama/Meta-Llama-3.1-8B-Instruct",
        "template": "llama3",
        "output_name": "Llama-3.1-8B-Instruct",
    },
}


@dataclass
class TrainMeta:
    adapter_path: Path
    model_name_or_path: str
    template: str
    finetuning_type: str
    dataset: str
    profile: str | None


def _sanitize(text: str) -> str:
    text = re.sub(r"[^a-zA-Z0-9]+", "_", text.strip())
    text = re.sub(r"_+", "_", text).strip("_")
    return text.lower()


def _model_short(model_name_or_path: str) -> str:
    lowered = model_name_or_path.lower()
    # Keep this mapping aligned with profiles in train_llamafactory.py.
    # Unknown models will fall back to sanitized repo/model name.
    mapping = {
        "meta-llama/meta-llama-3.1-8b-instruct": "llama3_1_8b",
        "qwen/qwen3-14b": "qwen3_14b",
        "qwen/qwen3-8b": "qwen3_8b",
        "qwen/qwen3-4b": "qwen3_4b",
        "qwen/qwen3-1.7b": "qwen3_1_7b",
        "qwen/qwen3-0.6b": "qwen3_0_6b",
        "google/gemma-3-4b-it": "gemma_3_4b",
        "google/gemma-3-1b-it": "gemma_3_1b",
        "google/gemma-3-270m-it": "gemma_3_270m",
        "google/gemma-3-27b-it": "gemma_3_27b",
        "google/gemma-3-12b-it": "gemma_3_12b",
    }
    if lowered in mapping:
        return mapping[lowered]
    return _sanitize(model_name_or_path.split("/")[-1])


def _run(
    cmd: list[str],
    env: dict[str, str] | None = None,
    conda_env: str | None = None,
    shell_init: str = "~/.bashrc",
) -> None:
    rendered = " ".join(shlex.quote(c) for c in cmd)
    if conda_env:
        shell_init_path = os.path.expanduser(shell_init)
        shell_cmd = (
            "set -euo pipefail; "
            f"source {shlex.quote(shell_init_path)}; "
            f"conda activate {shlex.quote(conda_env)}; "
            f"{rendered}"
        )
        print("[CMD]", f"bash -lc {shlex.quote(shell_cmd)}", flush=True)
        subprocess.run(["bash", "-lc", shell_cmd], check=True, env=env)
        return

    print("[CMD]", rendered, flush=True)
    subprocess.run(cmd, check=True, env=env)


def _dict_to_cli_args(args_dict: dict[str, Any]) -> list[str]:
    cli: list[str] = []
    for key, value in args_dict.items():
        flag = f"--{key}"
        if value is None:
            continue
        if isinstance(value, bool):
            if value:
                cli.append(flag)
            continue
        if isinstance(value, list):
            for item in value:
                cli.extend([flag, str(item)])
            continue
        cli.extend([flag, str(value)])
    return cli


def _latest_train_dir(lora_root: Path) -> Path:
    candidates = [p for p in lora_root.glob("train_*") if p.is_dir()]
    if not candidates:
        raise FileNotFoundError(f"No train_* directories found under {lora_root}")
    candidates.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    return candidates[0]


def _infer_train_meta(train_cfg: dict[str, Any], script_dir: Path) -> TrainMeta:
    profile = train_cfg.get("profile")
    finetuning_type = str(train_cfg.get("finetuning_type", "lora"))
    dataset = str(train_cfg["dataset"])

    if profile:
        if profile not in PROFILE_CFG:
            raise ValueError(f"Unknown profile: {profile}")
        p_cfg = PROFILE_CFG[profile]
        output_root = Path(train_cfg.get("output_root") or (script_dir / "saves")).resolve()
        lora_root = output_root / p_cfg["output_name"] / "lora"
        adapter_path = _latest_train_dir(lora_root)
        return TrainMeta(
            adapter_path=adapter_path,
            model_name_or_path=p_cfg["model"],
            template=p_cfg["template"],
            finetuning_type="lora",
            dataset=dataset,
            profile=profile,
        )

    if "model_name_or_path" not in train_cfg:
        raise ValueError("train.model_name_or_path is required when train.profile is not set")
    if "output_dir" not in train_cfg:
        raise ValueError("train.output_dir is required when train.profile is not set")
    template = str(train_cfg.get("template", "default"))
    adapter_path = Path(train_cfg["output_dir"]).resolve()
    return TrainMeta(
        adapter_path=adapter_path,
        model_name_or_path=str(train_cfg["model_name_or_path"]),
        template=template,
        finetuning_type=finetuning_type,
        dataset=dataset,
        profile=None,
    )


def _adapter_train_stamp(adapter_path: Path) -> tuple[str, str]:
    # train_2026-05-01-16-11-30 -> (2026_05_01, 2026_05_01_16_11_30)
    m = re.match(r"train_(\d{4})-(\d{2})-(\d{2})-(\d{2})-(\d{2})-(\d{2})$", adapter_path.name)
    if not m:
        return ("unknown_date", _sanitize(adapter_path.name))
    y, mo, d, hh, mm, ss = m.groups()
    return (f"{y}_{mo}_{d}", f"{y}_{mo}_{d}_{hh}_{mm}_{ss}")


def _resolve_merged_dir(train_meta: TrainMeta, merge_cfg: dict[str, Any]) -> Path:
    merged_root = Path(
        merge_cfg.get("merged_root") or (train_meta.adapter_path.parent / "merged_models")
    ).resolve()
    merged_root.mkdir(parents=True, exist_ok=True)

    if merge_cfg.get("name"):
        name = str(merge_cfg["name"])
    else:
        date_only, date_time = _adapter_train_stamp(train_meta.adapter_path)
        name_template = merge_cfg.get("name_template", "{model_short}_{date}_{dataset}")
        ctx = {
            "model_short": _model_short(train_meta.model_name_or_path),
            "dataset": _sanitize(train_meta.dataset),
            "date": date_only,
            "datetime": date_time,
            "profile": _sanitize(train_meta.profile or "custom"),
            "adapter": _sanitize(train_meta.adapter_path.name),
        }
        try:
            name = str(name_template).format(**ctx)
        except KeyError as exc:
            raise ValueError(f"Invalid placeholder in merge.name_template: {exc}") from exc
    return merged_root / name


def _run_train(config: dict[str, Any], script_dir: Path, runtime_cfg: dict[str, Any]) -> TrainMeta:
    train_cfg = dict(config.get("train", {}))
    if not train_cfg:
        raise ValueError("Missing required config section: train")
    if "dataset" not in train_cfg:
        raise ValueError("train.dataset is required")

    train_script = script_dir / "train_llamafactory.py"
    cmd = ["python", str(train_script)] + _dict_to_cli_args(train_cfg)
    _run(
        cmd,
        conda_env=runtime_cfg.get("train_conda_env"),
        shell_init=str(runtime_cfg.get("shell_init", "~/.bashrc")),
    )

    meta = _infer_train_meta(train_cfg, script_dir)
    if not meta.adapter_path.exists():
        raise FileNotFoundError(f"Adapter path not found after training: {meta.adapter_path}")
    return meta


def _run_merge(train_meta: TrainMeta, merge_cfg: dict[str, Any], runtime_cfg: dict[str, Any]) -> Path:
    merged_dir = _resolve_merged_dir(train_meta, merge_cfg)
    cmd = [
        "llamafactory-cli",
        "export",
        "--model_name_or_path",
        train_meta.model_name_or_path,
        "--adapter_name_or_path",
        str(train_meta.adapter_path),
        "--template",
        train_meta.template,
        "--finetuning_type",
        train_meta.finetuning_type,
        "--export_dir",
        str(merged_dir),
    ]

    export_device = merge_cfg.get("export_device", "auto")
    if export_device is not None:
        cmd += ["--export_device", str(export_device)]

    export_size = merge_cfg.get("export_size", 2)
    if export_size is not None:
        cmd += ["--export_size", str(export_size)]

    legacy = bool(merge_cfg.get("export_legacy_format", False))
    cmd += ["--export_legacy_format", str(legacy)]

    extra_export_args = merge_cfg.get("extra_export_args", "")
    if isinstance(extra_export_args, str) and extra_export_args.strip():
        cmd += shlex.split(extra_export_args)
    elif isinstance(extra_export_args, list):
        cmd += [str(x) for x in extra_export_args]

    _run(
        cmd,
        conda_env=runtime_cfg.get("merge_conda_env"),
        shell_init=str(runtime_cfg.get("shell_init", "~/.bashrc")),
    )
    return merged_dir


def _run_infer_worker(
    gpu: int,
    repeats: int,
    base_run_group: str,
    append_gpu_rep: bool,
    infer_cmd_base: list[str],
    conda_env: str | None,
    shell_init: str,
) -> None:
    for rep in range(1, repeats + 1):
        run_group = base_run_group
        if append_gpu_rep:
            run_group = f"{base_run_group}_gpu{gpu}_rep{rep}"
        cmd = infer_cmd_base + ["--run_group", run_group]
        env = os.environ.copy()
        env["CUDA_VISIBLE_DEVICES"] = str(gpu)
        _run(cmd, env=env, conda_env=conda_env, shell_init=shell_init)


def _run_infer(
    config: dict[str, Any],
    merged_dir: Path,
    project_root: Path,
    model_name_or_path: str,
    runtime_cfg: dict[str, Any],
) -> None:
    infer_cfg = dict(config.get("infer", {}))
    if not infer_cfg:
        return
    if not infer_cfg.get("enabled", True):
        return

    dataset = infer_cfg.get("dataset")
    corpus = infer_cfg.get("corpus")
    if not dataset or not corpus:
        raise ValueError("infer.dataset and infer.corpus are required when infer is enabled")

    gpus = infer_cfg.get("gpus", [0])
    if isinstance(gpus, int):
        gpus = [gpus]
    if not isinstance(gpus, list) or len(gpus) == 0:
        raise ValueError("infer.gpus must be a non-empty list (or a single int)")

    repeats = int(infer_cfg.get("repeats", 1))
    if repeats < 1:
        raise ValueError("infer.repeats must be >= 1")
    default_run_group = f"{_model_short(model_name_or_path)}_{_sanitize(str(dataset))}"
    base_run_group = str(infer_cfg.get("run_group", default_run_group))
    append_gpu_rep = bool(infer_cfg.get("append_gpu_rep_to_run_group", True))

    infer_script = project_root / "src" / "run_inference.py"
    infer_python = infer_cfg.get("python_executable")
    if infer_python:
        python_cmd = str(infer_python)
    else:
        python_cmd = "python"
    infer_cmd_base = [
        python_cmd,
        str(infer_script),
        "--model",
        str(merged_dir),
        "--dataset",
        str(dataset),
        "--corpus",
        str(corpus),
        "--temperature",
        str(infer_cfg.get("temperature", 0)),
        "--max_tokens",
        str(infer_cfg.get("max_tokens", 4096)),
        "--max_model_len",
        str(infer_cfg.get("max_model_len", 8192)),
    ]

    # Optional passthrough args
    for opt in ["task", "split", "seed", "batch_size", "top_p", "dtype", "tensor_parallel_size"]:
        if infer_cfg.get(opt) is not None:
            infer_cmd_base += [f"--{opt}", str(infer_cfg[opt])]

    extra_infer_args = infer_cfg.get("extra_infer_args", "")
    if isinstance(extra_infer_args, str) and extra_infer_args.strip():
        infer_cmd_base += shlex.split(extra_infer_args)
    elif isinstance(extra_infer_args, list):
        infer_cmd_base += [str(x) for x in extra_infer_args]

    with ThreadPoolExecutor(max_workers=len(gpus)) as ex:
        futures = [
            ex.submit(
                _run_infer_worker,
                int(gpu),
                repeats,
                base_run_group,
                append_gpu_rep,
                infer_cmd_base,
                runtime_cfg.get("infer_conda_env"),
                str(runtime_cfg.get("shell_init", "~/.bashrc")),
            )
            for gpu in gpus
        ]
        for fut in futures:
            fut.result()


def main() -> None:
    parser = argparse.ArgumentParser(description="One-click train -> merge -> infer pipeline.")
    parser.add_argument("--config", required=True, help="Path to JSON config")
    args = parser.parse_args()

    cfg_path = Path(args.config).resolve()
    config = json.loads(cfg_path.read_text())

    script_dir = Path(__file__).resolve().parent
    project_root = script_dir.parent

    print(f"[INFO] Config: {cfg_path}")
    runtime_cfg = dict(config.get("runtime", {}))

    train_meta = _run_train(config, script_dir, runtime_cfg)
    print(f"[INFO] Adapter path: {train_meta.adapter_path}")

    merge_cfg = dict(config.get("merge", {}))
    if merge_cfg.get("enabled", True):
        merged_dir = _run_merge(train_meta, merge_cfg, runtime_cfg)
        print(f"[INFO] Merged model dir: {merged_dir}")
    else:
        merged_dir = train_meta.adapter_path
        print(f"[INFO] Merge disabled. Using adapter path as model path: {merged_dir}")

    _run_infer(config, merged_dir, project_root, train_meta.model_name_or_path, runtime_cfg)
    print("[INFO] Pipeline completed.")


if __name__ == "__main__":
    main()
