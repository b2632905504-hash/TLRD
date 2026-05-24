# TLRD

This repository contains the code and data files used for tabular corpus
construction, teacher-model corpus generation, and LoRA finetuning.

Part of the corpus for finetuning is submitted in the Data attachment
for this ARR submission. 

The current code supports all datasets in our paper:

- `adult`
- `california`
- `diamonds`
- `home_credit`
- `okcupid_stem`
- `diabetes_130US`

We may not have permission to upload all raw dataset files with the anonymous
release. For datasets without a shipped CSV, download the
source data listed in `data/README.md` and place each file at the path declared
in `data/dataset.py` before running preprocessing.

## Repository Layout

```text
TLRD/
|-- data/                     datasets, preprocessing, corpus builders, and dataset source notes
|-- src/                      vLLM inference and teacher-model helpers
|-- finetune_llama_factory/   finetuning and train-merge-infer wrappers
|-- utils/                    shared utilities
`-- README.md
```

## Workflow Overview

The workflow below follows the code layout:

1. Configure the basic dataset information under `data/<dataset>/`.
2. Build prompts with `data/corpus_builder.py`. Use the default mode for
   base-model inference, `--mode finetune` to build the teacher-model input
   (the Tri-Level Augmented Input), or `--mode reasoning` for inference after
   TLRD finetuning.
3. Generate the finetuning corpus with
   `data/corpus_builder_finetune.py` in the vLLM environment.
4. Register the generated JSON in
   `finetune_llama_factory/dataset_info/dataset_info.json`.
5. Finetune with `finetune_llama_factory/train_llamafactory.py` in the
   LLaMA-Factory environment.
6. Optionally use `finetune_llama_factory/train_merge_infer.py` to run
   finetuning, adapter merge, and inference from one JSON config.

## Add a Dataset

To use a new tabular dataset with the corpus builders and inference code,
configure its CSV path, target column, task type, feature columns, prompt
instructions, and output parsing logic.

1. Add a dataset folder under `data/<dataset-name>/` with the expected local
   CSV path and a preprocessing script that writes `split_<seed>.pth` with
   train, validation, and test indices. The supported datasets provide examples
   under `data/`.
2. Add a `DatasetConfig` entry in `data/dataset.py`. Set the CSV path, target
   column, task type, and the categorical and numerical feature lists.
3. Implement a `TabularDataset` subclass in
   `data/<dataset-name>/<dataset-name>_dataset.py`. The class should define the
   prompt instructions needed by the workflow, implement `build_prompt`, and
   provide output parsing/target conversion methods used for evaluation.
4. Register that dataset class in `data/dataset_registry.py` so
   `data/corpus_builder.py` and `data/corpus_builder_finetune.py` can find it.
5. Register the same class in `src/run_inference.py` when the dataset should be
   available through the inference CLI.
6. Run the preprocessing script for the desired seed, then build corpora with
   `data/corpus_builder.py`.

For TLRD corpus generation, the dataset class should provide the instruction
variants used by the retained modes: a base inference instruction, a reasoning
instruction for `--mode reasoning`, a finetuning instruction for
`--mode finetune`, and a normal-finetuning instruction only if
`--mode normal_finetune` is needed.

After generating `split_<seed>.pth`, run a small corpus build to check that the
new dataset is connected:

```bash
python data/corpus_builder.py \
  --dataset <dataset-name> \
  --split train \
  --seed 42 \
  --limit 10
```

## Recommended Environments

Use two separate environments for the workflow.

1. **vLLM environment**

   Use this environment for dataset preprocessing, prompt corpus construction,
   inference, and teacher-model generation of finetuning corpora.

   Main entry points:

   - `data/corpus_builder.py`
   - `data/corpus_builder_finetune.py`
   - `src/run_inference.py`

2. **LLaMA-Factory environment**

   Use this environment for supervised finetuning and optional adapter export.
   The training wrapper calls `llamafactory-cli`.

   Main entry point:

   - `finetune_llama_factory/train_llamafactory.py`

Keeping these environments separate is recommended because vLLM inference and
LLaMA-Factory training may require different dependency versions.

## Workflow

### 1. Build dataset splits

Run the dataset preprocessing script before building corpora when the required
`split_<seed>.pth` file is not already present.

Run each preprocessing script from its dataset directory so its relative CSV
path and generated split path resolve locally. See `data/README.md` for raw
dataset download locations and dataset-specific notes.

```bash
(cd data/adult && python preprocess.py --seed 42)
(cd data/california && python preprocess.py --seed 42)
(cd data/diamonds && python preprocess.py --seed 42)
(cd data/home_credit && python preprocess.py --seed 42)
(cd data/okcupid_stem && python preprocess.py --seed 42)
```

### 2. Build a prompt corpus

In the vLLM environment, build an Alpaca-style corpus from a dataset split.
The corpus builder is used both for base-model inference corpora and for the
Tri-Level Augmented Input corpora that are passed to the teacher-generation
script.

```bash
python data/corpus_builder.py \
  --dataset adult \
  --split train \
  --mode finetune \
  --with-stats \
  --with-comparison
```

By default, generated corpora are written under the selected dataset directory.

Important `data/corpus_builder.py` arguments:

- `--dataset`: dataset name. This submission supports all datasets listed in
  `data/README.md`.
- `--split`: split to build: `train`, `valid`, `test`, or `all`.
- `--seed`: split seed used to load `split_<seed>.pth`.
- `--task`: optional task choice for datasets that expose multiple task views,
  such as California housing.
- `--limit`: optional sample cap for quick runs or controlled corpus sizes.
- `--output-dir`, `--output-name`, and `--no-timestamp`: output location and
  naming controls.
- `--with-stats`: add training-set statistics to prompts.
- `--inline-feature-stats`, `--inline-header-block`, `--per-class-stats`, and
  related stats options: control how statistics are rendered.
- `--with-comparison`: add retrieved similar training cases to prompts.
- `--comparison-n`, `--comparison-position`, and `--balance-comparison`:
  control the retrieved comparison examples.

The paper configuration uses training-set statistics with per-class statistics
and balanced retrieved comparison examples placed after the query:

```bash
--with-stats \
--per-class-stats \
--with-comparison \
--comparison-position after_query \
--balance-comparison
```

Corpus builder modes:

- Default mode, `zeroshot`: build prompts for inference with the base model.
  Omit `--mode` or pass `--mode zeroshot`.
- `finetune`: build prompts used by
  `data/corpus_builder_finetune.py` to generate the distilled finetuning
  corpus. This mode uses the dataset finetuning instruction and appends the
  given prediction label to the prompt input for teacher reasoning.
- `reasoning`: build inference prompts for the model after TLRD finetuning.
  This mode uses the dataset reasoning instruction.
- `normal_finetune`: an alternative finetuning instruction variant retained by
  the corpus builder.

### 3. Generate a finetuning corpus with vLLM

Use `data/corpus_builder_finetune.py` with a vLLM teacher model to turn an
existing `--mode finetune` prompt corpus into a finetuning corpus.

```bash
python data/corpus_builder_finetune.py \
  --dataset adult \
  --corpus-path data/adult/corpus/<corpus-file>.json \
  --teacher vllm \
  --model openai/gpt-oss-20b \
  --tensor-parallel-size 1
```

The generated finetuning JSON is saved next to the explicit input corpus path,
or under `data/<dataset>/corpus_finetune/` when a dataset-level corpus search is
used.

For base-model inference, build a default `zeroshot` corpus and pass it to
`src/run_inference.py`. For post-finetuning TLRD inference, build a
`--mode reasoning` corpus and use that corpus with the finetuned or merged
model.

### 4. Register the finetuning JSON for LLaMA-Factory

Before training, add the generated JSON file to
`finetune_llama_factory/dataset_info/dataset_info.json` using the
LLaMA-Factory dataset registry format. The dataset name used in that registry
is the value passed to `--dataset` during finetuning.

For example, a generated Alpaca-style finetuning corpus can be registered with
a relative path like this:

```json
{
  "adult_tlrd_finetune": {
    "file_name": "../../data/adult/corpus_finetune/<finetuning-corpus-file>.json",
    "formatting": "alpaca"
  }
}
```

In this example, `adult_tlrd_finetune` is the LLaMA-Factory dataset name passed
to `--dataset`, and `file_name` points from the dataset-info directory to the
generated finetuning JSON.

### 5. Finetune in the LLaMA-Factory environment

Run the training wrapper from the LLaMA-Factory environment.

```bash
python finetune_llama_factory/train_llamafactory.py \
  --profile qwen3-8b \
  --dataset <llamafactory-dataset-name> \
  --cutoff_len 6500
```

The wrapper provides predefined profiles and writes timestamped training outputs
under `finetune_llama_factory/saves/` unless another output root is supplied.

## One-Stop Train, Merge, and Inference

`finetune_llama_factory/train_merge_infer.py` is a JSON-configured pipeline for
running three stages together:

1. call `train_llamafactory.py` to finetune;
2. export a merged model with `llamafactory-cli export`;
3. optionally call `src/run_inference.py` on one or more GPUs, with repeated
   inference runs if requested.

Run it with:

```bash
python finetune_llama_factory/train_merge_infer.py \
  --config <pipeline-config>.json
```

The JSON config uses these sections:

- `runtime`: optional conda environment names for the train, merge, and
  inference stages. This lets the pipeline keep LLaMA-Factory training and
  vLLM inference in separate environments.
- `train`: arguments forwarded to `train_llamafactory.py`, including the
  finetuning profile and the LLaMA-Factory dataset name.
- `merge`: export settings for the merged model directory and export options.
- `infer`: optional inference settings, including the inference dataset, corpus
  path, GPU list, and repeat count.

Minimal config shape:

```json
{
  "runtime": {
    "train_conda_env": "<llamafactory-env>",
    "merge_conda_env": "<llamafactory-env>",
    "infer_conda_env": "<vllm-env>"
  },
  "train": {
    "profile": "qwen3-8b",
    "dataset": "<llamafactory-dataset-name>",
    "cutoff_len": 6500
  },
  "merge": {
    "merged_root": "finetune_llama_factory/merged_models"
  },
  "infer": {
    "enabled": true,
    "dataset": "adult",
    "corpus": "data/adult/corpus/<reasoning-corpus-file>.json",
    "gpus": [0],
    "repeats": 1
  }
}
```

When the inference corpus has already been built with
`data/corpus_builder.py --mode reasoning`, the inference script reads the
reasoning prompt directly from that corpus file.
