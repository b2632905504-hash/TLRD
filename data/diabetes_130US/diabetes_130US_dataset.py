"""
Diabetes 130-US Dataset Class with LLM Prompt Building Support.

Multiclass classification:
    NO  -> no readmission
    >30 -> readmitted after 30 days
    <30 -> readmitted within 30 days
"""

import re
from typing import Optional, Literal

from data.dataset import TabularDataset


class Diabetes130USDataset(TabularDataset):
    """Diabetes 130-US hospital readmission dataset for multiclass classification."""

    LABELS = ["NO", ">30", "<30"]
    LABEL_TO_ID = {label: i for i, label in enumerate(LABELS)}
    ID_TO_LABEL = {i: label for label, i in LABEL_TO_ID.items()}

    INSTRUCTION = (
        "You are a hospital readmission classifier. "
        "Based on the following patient encounter features, predict the readmission category for this patient. "
        "Please explain your reasoning.\n\n"
        "At the very end of your reasoning, on a separate line, output exactly:\n"
        "Prediction is <value>.\n"
        "where <value> must be one of: NO, >30, <30.\n"
        "NO means no readmission; >30 means readmitted after 30 days; <30 means readmitted within 30 days.\n"
        "Do not add any other text after this line.\n\n"
    )

    INSTRUCTION_REASONING = (
        "You are a hospital readmission classifier. Based on the following patient encounter features, conduct in-depth qualitative analysis to predict the readmission category for this patient. "
        "You should use dataset statistics and similar historical cases to assist your reasoning. Please explain your reasoning in the following way.\n\n"
        "Reasoning format:\n"
        "(1). Self feature-based reasoning:\n"
        "(2). Global statistics-based reasoning:\n"
        "(3). Similar-cases reasoning:\n"
        "(4). Final Resolution:\n"
        "At the very end of your reasoning, on a separate line, output exactly:\n"
        "Prediction is <value>.\n"
        "where <value> must be one of: NO, >30, <30.\n"
        "NO means no readmission; >30 means readmitted after 30 days; <30 means readmitted within 30 days.\n"
        "Do not add any other text after this line.\n\n"
    )

    INSTRUCTION_FINETUNE = (
        "You are a hospital readmission classifier. Based on the following patient encounter features, conduct in-depth qualitative analysis to explain why THE GIVEN PREDICTION LABEL for this patient's readmission category is plausible (valid labels are: NO, >30, <30; NO means no readmission, >30 means readmitted after 30 days, <30 means readmitted within 30 days). You are also provided with dataset statistics and similar historical cases to assist your reasoning. Please explain your reasoning in the following way.\n\n"
        "Reasoning format:\n"
        "(1). Self feature-based reasoning:\n"
        "- Analyze each feature from \"Current Sample Features\" strictly using general clinical logic and hospital utilization common sense.\n"
        "Execution Rules:\n"
        "- Briefly explain your intuition, and state its directional impact on readmission risk and timing.\n"
        "- Focus on the self feature logical connection.\n"
        "Prohibitions:\n"
        "- DO NOT use any dataset-level quantities (e.g., mean, percentile) from \"Dataset Statistics\" or historical cases in this step.\n\n"
        "(2). Global statistics-based reasoning:\n"
        "- Contextualize the sample by comparing its key feature values against the provided dataset statistics for EACH class.\n"
        "- Prioritize those with the clearest class separation or most extreme placement for this sample.\n"
        "- For each feature, explicitly cite the relevant statistics (quote the numbers) when stating where the sample falls relative to the stats (e.g., closer to which class mean/median, or which percentile), highlight deviations/alignments between the sample and the statistical benchmarks, then explain the directional implication.\n\n"
        "(3). Similar-cases reasoning: Use the provided similar historical cases to extract feature-combination alignment patterns and critical deviations, to further strengthen and validate your prediction logic.\n"
        "Execution Rules:\n"
        "- Extract consistent feature-combination alignment pattern from similar cases. (e.g., [A + B + C] matches).\n"
        "- Identify critical feature deviations between the current sample and similar cases  (e.g., [D] differs).\n"
        "- For every pattern or deviation mentioned, explicitly cite the current sample's corresponding features.\n"
        "- Clearly explain how the current sample's features align with patterns or diverge from deviations.\n"
        "Prohibitions:\n"
        "- Do NOT reference historical cases using numeric identifiers (IDs, order/rank, similarity scores).\n"
        "- Do NOT use the words \"Example\", \"record\", or \"case\".\n"
        "- Structure your response in two sections:\n"
        "1. <Patterns>: List all consistent feature-combination alignment patterns, each explicitly linked to the current sample's matching features.\n"
        "2. <Deviations>: List the most critical feature deviations, each explicitly linked to the current sample's differing features\n\n"
        "(4). Final Resolution: Weigh the evidence from the above steps.\n"
        "If there are conflicts, explicitly explain how you resolve them.\n\n"
        "Use a layered, step-by-step reasoning chain: start from feature logic, then refine with statistics, then validate with similar cases.\n"
        "At the very end of your reasoning, on a separate line, output exactly:\n"
        "Prediction is <value>.\n"
        "where <value> must be one of: NO, >30, <30.\n"
        "NO means no readmission; >30 means readmitted after 30 days; <30 means readmitted within 30 days.\n"
        "Do not add any other text after this line.\n\n"
    )

    INSTRUCTION_NORMAL_FINETUNE = (
        "You are a hospital readmission classifier. Based on the following patient encounter features, explain why THE GIVEN PREDICTION LABEL for this patient's readmission category is plausible (valid labels are: NO, >30, <30; NO means no readmission, >30 means readmitted after 30 days, <30 means readmitted within 30 days).\n\n"
        "At the very end of your reasoning, on a separate line, output exactly:\n"
        "Prediction is <value>.\n"
        "where <value> must be one of: NO, >30, <30.\n"
        "NO means no readmission; >30 means readmitted after 30 days; <30 means readmitted within 30 days.\n"
        "Do not add any other text after this line.\n\n"
    )

    INSTRUCTION_WITH_STATS = (
        "You are a hospital readmission classifier. "
        "Based on the dataset statistics and patient encounter features below, predict the readmission category. "
        "Answer using one of: NO, >30, <30 "
        "(NO means no readmission, >30 means readmitted after 30 days, <30 means readmitted within 30 days). "
        "Please explain your reasoning.\n\n"
    )

    FEATURE_DESCRIPTIONS = {
        "discharge_disposition_id": "Discharge destination/status code from this encounter.",
        "number_inpatient": "Number of inpatient admissions in the previous year.",
        "number_diagnoses": "Total number of diagnosis codes recorded in this encounter.",
        "diag_1": "Primary diagnosis group/category for this encounter.",
        "age": "Patient age group bucket (<30, 30-60, >60 in this processed dataset).",
        "diabetesMed": "Whether any diabetes medication was prescribed (Yes/No).",
        "medical_specialty": "Admitting physician specialty (or 'missing' when unknown).",
        "time_in_hospital": "Length of stay for this encounter in days.",
        "num_lab_procedures": "Number of laboratory procedures performed during this encounter.",
        "admission_source_id": "Admission source code (for example physician referral, ER, transfer).",
        "num_medications": "Number of distinct medications prescribed during this encounter.",
        "admission_type_id": "Admission type code (for example emergency, urgent, elective).",
        "number_outpatient": "Number of outpatient visits in the previous year.",
        "num_procedures": "Number of non-laboratory procedures performed during this encounter.",
        "diag_2": "Secondary diagnosis group/category for this encounter.",
    }

    def __init__(
        self,
        split: Optional[Literal["train", "valid", "test"]] = None,
        seed: int = 42,
        split_path: Optional[str] = None,
        transform: Optional[callable] = None,
    ):
        super().__init__(
            dataset_name="diabetes_130US",
            split=split,
            seed=seed,
            split_path=split_path,
            transform=transform,
        )

    @classmethod
    def _map_label(cls, value: str) -> Optional[str]:
        if value is None:
            return None
        s = str(value).strip()
        if s in cls.LABEL_TO_ID:
            return s
        if s.lower() == "no":
            return "NO"
        return None

    def _build_feature_explanations(self) -> str:
        explanations = []
        for col in self.FEATURE_DESCRIPTIONS:
            if col in self.feature_cols:
                explanations.append(f"- {col}: {self.FEATURE_DESCRIPTIONS[col]}")
        if explanations:
            return "Feature explanations & Feature importance (high to low):\n" + "\n".join(explanations)
        return ""

    def build_prompt(self, idx: int) -> dict:
        sample = self[idx]
        features = sample["features"]
        target = sample["target"]

        lines = []
        ordered_cols = [c for c in self.FEATURE_DESCRIPTIONS if c in self.feature_cols]
        ordered_cols.extend([c for c in self.feature_cols if c not in ordered_cols])
        for col in ordered_cols:
            value = features[col]
            if value == "missing" or (isinstance(value, float) and str(value) == "nan"):
                val_str = "missing"
            elif value == "?":
                val_str = "missing"
            else:
                val_str = str(value)
            lines.append(f"{col} is {val_str}")
        input_text = ". ".join(lines)

        answer = self.target_to_output(target)

        feature_explanations = self._build_feature_explanations()
        if feature_explanations:
            instruction = self.INSTRUCTION.rstrip() + "\n\n" + feature_explanations + "\n"
            instruction_reasoning = (
                self.INSTRUCTION_REASONING.rstrip() + "\n\n" + feature_explanations + "\n"
            )
        else:
            instruction = self.INSTRUCTION
            instruction_reasoning = self.INSTRUCTION_REASONING

        return {
            "instruction": instruction,
            "instruction_reasoning": instruction_reasoning,
            "input": input_text,
            "output": answer,
        }

    def build_all_prompts(self) -> list:
        return [self.build_prompt(i) for i in range(len(self))]

    @classmethod
    def parse_output(cls, raw_text: str) -> Optional[str]:
        text = raw_text.strip().lower()

        line_pred_matches = re.findall(
            r"^\s*\**\s*prediction\s+is\s+\**\s*([a-zA-Z0-9_\-<>\s]+)\b",
            text,
            flags=re.IGNORECASE | re.MULTILINE,
        )
        for candidate in reversed(line_pred_matches):
            norm = cls._map_label(candidate)
            if norm is not None:
                return norm

        any_pred_matches = re.findall(
            r"\**\s*prediction\s+is\s+\**\s*([a-zA-Z0-9_\-<>\s]+)\b",
            text,
            flags=re.IGNORECASE,
        )
        for candidate in reversed(any_pred_matches):
            norm = cls._map_label(candidate)
            if norm is not None:
                return norm

        return None

    @classmethod
    def output_to_target(cls, output: str) -> Optional[int]:
        norm = cls._map_label(output)
        if norm is None:
            return None
        return cls.LABEL_TO_ID[norm]

    @classmethod
    def target_to_output(cls, target) -> str:
        if isinstance(target, str):
            s = target.strip()
            norm = cls._map_label(s)
            if norm is not None:
                return norm
            return "NO"
        try:
            idx = int(target)
        except (TypeError, ValueError):
            return "NO"
        return cls.ID_TO_LABEL.get(idx, "NO")


def get_diabetes_130us_dataset(
    split: Optional[Literal["train", "valid", "test"]] = None,
    seed: int = 42,
) -> Diabetes130USDataset:
    return Diabetes130USDataset(split=split, seed=seed)


def convert_target_to_text(y, sample: Optional[dict] = None, metadata: Optional[dict] = None) -> str:
    del sample, metadata
    label = Diabetes130USDataset.target_to_output(y)
    if label == "<30":
        return "This patient is readmitted within 30 days."
    if label == ">30":
        return "This patient is readmitted after 30 days."
    return "This patient has no readmission."
