"""
Home Credit Dataset Class with LLM Prompt Building Support.

Binary classification:
    TARGET = 1 -> likely to default on the loan
    TARGET = 0 -> likely to repay
"""

import re
from typing import Optional, Literal

from data.dataset import TabularDataset


class HomeCreditDataset(TabularDataset):
    """
    Home Credit Default Risk dataset for binary classification.

    Task: Predict whether the applicant will default on the loan.
    Target: TARGET (1 = default, 0 = non-default)
    """

    INSTRUCTION = (
        "You are a credit risk classifier. "
        "Based on the following applicant and loan features, predict whether the applicant will default on the loan. "
        "Please explain your reasoning.\n\n"
        "At the very end of your reasoning, on a separate line, output exactly:\n"
        "Prediction is <value>.\n"
        "where <value> is either yes or no.\n"
        "yes means the applicant is likely to default; no means the applicant is not likely to default.\n"
        "Do not add any other text after this line.\n\n"
    )

    INSTRUCTION_REASONING = (
        "You are a credit risk classifier. Based on the following applicant and loan features, conduct in-depth qualitative analysis to predict whether the applicant will default on the loan. "
        "You should use dataset statistics and similar historical cases to assist your reasoning. Please explain your reasoning in the following way.\n\n"
        "Reasoning format:\n"
        "(1). Self feature-based reasoning:\n"
        "(2). Global statistics-based reasoning:\n"
        "(3). Similar-cases reasoning:\n"
        "(4). Final Resolution:\n"
        "At the very end of your reasoning, on a separate line, output exactly:\n"
        "Prediction is <value>.\n"
        "where <value> is either yes or no.\n"
        "yes means the applicant is likely to default; no means the applicant is not likely to default.\n"
        "Do not add any other text after this line.\n\n"
    )

    INSTRUCTION_FINETUNE = (
        "You are a credit risk classifier. Based on the following applicant and loan features, conduct in-depth qualitative analysis to explain why THE GIVEN PREDICTION LABEL for this applicant's default risk is plausible (yes means the applicant is likely to default; no means the applicant is not likely to default). You are also provided with dataset statistics and similar historical cases to assist your reasoning. Please explain your reasoning in the following way.\n\n"
        "Reasoning format:\n"

        "(1). Self feature-based reasoning:\n"
        "- Analyze each feature from \"Current Sample Features\" strictly using general credit risk logic and lending common sense.\n"
        "Execution Rules:\n"
        "- Briefly explain your intuition, and state its directional impact on default risk.\n"
        "- Focus on the self feature logical connection.\n"
        "Prohibitions:\n"
        "- DO NOT use any dataset-level quantities (e.g., mean, percentile) from \"Dataset Statistics\" or historical cases in this step.\n\n"

        "(2). Global statistics-based reasoning:\n"
        "- Contextualize the sample by comparing its key feature values against the provided dataset statistics for BOTH classes.\n"
        "- Prioritize those with the clearest class separation or most extreme placement for this sample.\n"
        "- For each feature, explicitly cite the relevant statistics (quote the numbers) when stating where the sample falls relative to the stats (e.g., closer to which class mean/median, or which percentile), highlight deviations/alignments between the sample and the statistical benchmarks, then explain the directional implication.\n\n"

        "(3). Similar-cases reasoning: Use the provided similar historical cases to extract feature-combination alignment patterns and critical deviations, to further strengthen and validate your prediction logic.\n"
        "Execution Rules:\n"
        "- Extract consistent feature-combination alignment pattern from similar cases. (e.g., [A + B + C] matches).\n"
        "- Identify critical feature deviations between the current sample and similar cases  (e.g., [D] differs).\n"
        "- For every pattern or deviation mentioned, explicitly cite the current sample’s corresponding features.\n"
        "- Clearly explain how the current sample’s features align with patterns or diverge from deviations.\n"
        "Prohibitions:\n"
        "- Do NOT reference historical cases using numeric identifiers (IDs, order/rank, similarity scores).\n"
        "- Do NOT use the words \"Example\", \"record\", or \"case\".\n"
        "- Structure your response in two sections:\n"
        "1. <Patterns>: List all consistent feature-combination alignment patterns, each explicitly linked to the current sample’s matching features.\n"
        "2. <Deviations>: List the most critical feature deviations, each explicitly linked to the current sample’s differing features\n\n"

        "(4). Final Resolution: Weigh the evidence from the above steps.\n"
        "If there are conflicts, explicitly explain how you resolve them.\n\n"
        "Use a layered, step-by-step reasoning chain: start from feature logic, then refine with statistics, then validate with similar cases.\n"
        "At the very end of your reasoning, on a separate line, output exactly:\n"
        "Prediction is <value>.\n"
        "where <value> is either yes or no.\n"
        "yes means the applicant is likely to default; no means the applicant is not likely to default.\n"
        "Do not add any other text after this line.\n\n"
    )

    # Placeholder for the lighter normal finetune prompt variant.
    INSTRUCTION_NORMAL_FINETUNE = (
        "You are a credit risk classifier. Based on the following applicant and loan features, explain why THE GIVEN PREDICTION LABEL for this applicant's default risk is plausible (yes means the applicant is likely to default; no means the applicant is not likely to default).\n\n"
        "At the very end of your reasoning, on a separate line, output exactly:\n"
        "Prediction is <value>.\n"
        "where <value> is either yes or no.\n"
        "yes means the applicant is likely to default; no means the applicant is not likely to default.\n"
        "Do not add any other text after this line.\n\n"
    )

    INSTRUCTION_WITH_STATS = (
        "You are a credit risk classifier. "
        "Based on the dataset statistics and the applicant features below, "
        "predict whether the applicant will default on the loan. "
        "Answer 'yes' if the applicant is likely to default, or 'no' if the applicant is not likely to default. "
        "Please explain your reasoning.\n\n"
    )


    FEATURE_DESCRIPTIONS = {
        "EXT_SOURCE_3": "External risk score 3 (normalized).",
        "EXT_SOURCE_2": "External risk score 2 (normalized).",
        "AMT_GOODS_PRICE": "Price of goods for which the loan is requested.",
        "EXT_SOURCE_1": "External risk score 1 (normalized).",
        "AMT_CREDIT": "Loan credit amount requested.",
        "DAYS_EMPLOYED": "Days employed before application (relative days).",
        "CODE_GENDER": "Client gender.",
        "NAME_EDUCATION_TYPE": "Highest education level achieved.",
        "DAYS_BIRTH": "Client age in days at application (relative days).",
        "AMT_ANNUITY": "Loan annuity amount.",
        "OCCUPATION_TYPE": "Client occupation type.",
        "FLAG_OWN_CAR": "Whether the client owns a car (Y/N).",
        "DAYS_ID_PUBLISH": "Days since ID document update before application.",
        "ORGANIZATION_TYPE": "Type of organization where client works.",
        "FLAG_DOCUMENT_3": "Whether document #3 was provided (0/1).",
    }

    def __init__(
        self,
        split: Optional[Literal["train", "valid", "test"]] = None,
        seed: int = 42,
        split_path: Optional[str] = None,
        transform: Optional[callable] = None,
    ):
        """
        Args:
            split: Which split to load ('train', 'valid', 'test', or None for full)
            seed: Random seed used for split file (split_{seed}.pth)
            split_path: Custom path to split file (overrides default)
            transform: Optional transform to apply to samples
        """
        super().__init__(
            dataset_name="home_credit",
            split=split,
            seed=seed,
            split_path=split_path,
            transform=transform,
        )

    def _build_feature_explanations(self) -> str:
        """Build feature explanations text from FEATURE_DESCRIPTIONS."""
        explanations = []
        for col in self.FEATURE_DESCRIPTIONS:
            if col in self.feature_cols:
                explanations.append(f"- {col}: {self.FEATURE_DESCRIPTIONS[col]}")
        if explanations:
            return "Feature explanations & Feature importance (high to low):\n" + "\n".join(explanations)
        return ""

    def build_prompt(self, idx: int) -> dict:
        """
        Build an Alpaca-style prompt for the given sample index.

        Args:
            idx: Index of the sample in the current split

        Returns:
            dict with keys: instruction, input, output
        """
        sample = self[idx]
        features = sample["features"]
        target = sample["target"]

        # Build input text from features using original column names
        lines = []
        ordered_cols = [c for c in self.FEATURE_DESCRIPTIONS if c in self.feature_cols]
        ordered_cols.extend([c for c in self.feature_cols if c not in ordered_cols])
        for col in ordered_cols:
            value = features[col]

            if value == "missing" or (isinstance(value, float) and str(value) == "nan"):
                val_str = "missing"
            else:
                val_str = str(value)

            lines.append(f"{col} is {val_str}")

        input_text = ". ".join(lines)

        # Convert target to yes/no
        is_default = False
        if isinstance(target, str):
            is_default = target.strip().lower() in ("1", "yes", "default", "true")
        else:
            try:
                is_default = int(target) == 1
            except (TypeError, ValueError):
                is_default = False

        answer = "yes" if is_default else "no"

        # Append feature explanations to instruction
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
        """
        Build prompts for all samples in the current split.

        Returns:
            List of prompt dicts
        """
        return [self.build_prompt(i) for i in range(len(self))]

    @staticmethod
    def parse_output(raw_text: str) -> Optional[str]:
        """
        Parse yes/no from model output for Home Credit dataset.

        Args:
            raw_text: Raw model output text

        Returns:
            'yes', 'no', or None if cannot parse
        """
        text = raw_text.strip().lower()

        # Priority 1: Line-start "Prediction is <yes/no>." (allow ** and optional colon)
        line_pred_matches = re.findall(
            r"^\s*\**\s*prediction\s+is\s*\**\s*[:：]?\s*\**\s*(yes|no)\b",
            text,
            flags=re.IGNORECASE | re.MULTILINE,
        )
        if line_pred_matches:
            return line_pred_matches[-1].lower()

        # Priority 2: Line-start numeric 0/1 (allow ** and optional colon)
        line_num_matches = re.findall(
            r"^\s*\**\s*prediction\s+is\s*\**\s*[:：]?\s*\**\s*([01])\b",
            text,
            flags=re.IGNORECASE | re.MULTILINE,
        )
        if line_num_matches:
            return "yes" if line_num_matches[-1] == "1" else "no"

        # Priority 3: Anywhere "Prediction is <yes/no>" (allow ** and optional colon)
        any_pred_matches = re.findall(
            r"\**\s*prediction\s+is\s*\**\s*[:：]?\s*\**\s*(yes|no)\b",
            text,
            flags=re.IGNORECASE,
        )
        if any_pred_matches:
            return any_pred_matches[-1].lower()

        # Priority 4: Anywhere numeric 0/1 (allow ** and optional colon)
        any_num_matches = re.findall(
            r"\**\s*prediction\s+is\s*\**\s*[:：]?\s*\**\s*([01])\b",
            text,
            flags=re.IGNORECASE,
        )
        if any_num_matches:
            return "yes" if any_num_matches[-1] == "1" else "no"

        return None

    @staticmethod
    def output_to_target(output: str) -> int:
        """
        Convert output label to binary target.

        Args:
            output: 'yes' or 'no'

        Returns:
            1 for yes (default), 0 for no (non-default)
        """
        return 1 if output and output.lower() == "yes" else 0

    @staticmethod
    def target_to_output(target) -> str:
        """
        Convert binary target to output label.

        Args:
            target: 1 or 0, or 'yes' / 'no'

        Returns:
            'yes' or 'no'
        """
        if isinstance(target, str):
            return "yes" if target.strip().lower() in ("1", "yes", "true", "default") else "no"
        return "yes" if int(target) == 1 else "no"


def get_home_credit_dataset(
    split: Optional[Literal["train", "valid", "test"]] = None,
    seed: int = 42,
) -> HomeCreditDataset:
    """Get Home Credit dataset with specified split."""
    return HomeCreditDataset(split=split, seed=seed)


def convert_target_to_text(
    y,
    sample: Optional[dict] = None,
    metadata: Optional[dict] = None,
) -> str:
    """
    Convert target value to a natural-language sentence describing the ground-truth label.

    This function is used by comparison.py for RAG retrieval to generate
    human-readable label text for each retrieved training example.
    """
    is_default = False
    if isinstance(y, str):
        is_default = y.strip().lower() in ("1", "yes", "true", "default")
    else:
        try:
            is_default = int(y) == 1
        except (TypeError, ValueError):
            is_default = False

    if is_default:
        return "This applicant will default on the loan."
    return "This applicant will not default on the loan."


if __name__ == "__main__":
    # Quick test
    ds = HomeCreditDataset(split="train", seed=42)
    print(ds)
    print(f"\nClass distribution: {ds.get_class_distribution()}")
    prompt = ds.build_prompt(0)
    print(f"\nInstruction: {prompt['instruction'][:100]}...")
    print(f"Input: {prompt['input'][:200]}...")
    print(f"Output: {prompt['output']}")
