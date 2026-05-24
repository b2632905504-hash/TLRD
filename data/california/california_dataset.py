"""
California Housing Dataset Class with LLM Prompt Building Support.

The CSV path and target column are configured via `DATASET_REGISTRY`
in `data/dataset.py`. By default we assume a schema similar to the
sklearn California housing dataset, with target column `MedHouseVal`.
"""

import re
from typing import Optional, Literal, Union

import numpy as np

from data.dataset import TabularDataset, DatasetConfig, DATASET_REGISTRY


class CaliforniaDataset(TabularDataset):
    """
    California Housing Dataset.
    """

    # Instructions
    INSTRUCTION_REGRESSION = (
        "You are a house price estimator. "
        "Based on the following housing and geographic features in California, predict the median house value in US Dollars for this area. "
        "Please explain your reasoning.\n\n"
        "At the very end of your reasoning, on a separate line, output exactly:\n"
        "Prediction is <value>.\n"
        "where <value> is the predicted median house value in US Dollars as a numeric value.\n"
        "Do not add any other text after this line.\n\n"
    )

    INSTRUCTION_REASONING_REGRESSION = (
        "You are a house price estimator. Based on the following housing and geographic features in California, conduct in-depth qualitative analysis to predict the median house value in US Dollars for this area. "
        "You should use dataset statistics and similar historical cases to assist your reasoning. Please explain your reasoning in the following way.\n\n"
        "Reasoning format:\n"
        "(1). Self feature-based reasoning:\n"
        "(2). Global statistics-based reasoning across the value distribution:\n"
        "(3). Similar-cases reasoning:\n"
        "(4). Final Resolution:\n"
        "At the very end of your reasoning, on a separate line, output exactly:\n"
        "Prediction is <value>.\n"
        "where <value> is the predicted median house value in US Dollars as a numeric value.\n"
        "Do not add any other text after this line.\n\n"
    )

    INSTRUCTION_FINETUNE = (
        "You are a house price estimator. Based on the following housing and geographic features in California, conduct in-depth qualitative analysis to explain why THE GIVEN PREDICTION LABEL for this area's median house value in US Dollars is plausible. You are also provided with dataset statistics and similar historical cases to assist your reasoning. Please explain your reasoning in the following way.\n\n"
        "Reasoning format:\n"
        "(1). Self feature-based reasoning:\n"
        "- Analyze each feature from \"Current Sample Features\" strictly using general real estate logic and California housing market common sense.\n"
        "Execution Rules:\n"
        "- Briefly explain your intuition, and state its directional impact on house value.\n"
        "- Focus on the self feature logical connection.\n"
        "Prohibitions:\n"
        "- DO NOT use any dataset-level quantities (e.g., mean, percentile) from \"Dataset Statistics\" or historical cases in this step.\n\n"
        "(2). Global statistics-based reasoning:\n"
        "- Contextualize the sample by comparing its key feature values against the provided dataset statistics across the value distribution.\n"
        "- Prioritize features with the clearest value correlation or most extreme placement for this sample.\n"
        "- For each feature, explicitly cite the relevant statistics (quote the numbers) when stating where the sample falls relative to the stats (e.g., closer to which percentile, above/below mean/median), highlight deviations/alignments between the sample and the statistical benchmarks, then explain the directional implication.\n\n"
        "(3). Similar-cases reasoning: Use the provided similar historical cases to extract feature-combination alignment patterns and critical deviations, to further strengthen and validate your prediction logic.\n"
        "Execution Rules:\n"
        "- Extract consistent feature-combination alignment pattern from similar cases. (e.g., [A + B + C] matches).\n"
        "- Identify critical feature deviations between the current sample and similar cases (e.g., [D] differs).\n"
        "- For every pattern or deviation mentioned, explicitly cite the current sample's corresponding features.\n"
        "- Clearly explain how the current sample's features align with patterns or diverge from deviations.\n"
        "Prohibitions:\n"
        "- Do NOT reference historical cases using numeric identifiers (IDs, order/rank, similarity scores).\n"
        "- Do NOT use the words \"Example\", \"record\", or \"case\".\n"
        "- Structure your response in two sections:\n"
        "1. <Patterns>: List all consistent feature-combination alignment patterns, each explicitly linked to the current sample's matching features.\n"
        "2. <Deviations>: List the most critical feature deviations, each explicitly linked to the current sample's differing features.\n\n"
        "(4). Final Resolution: Weigh the evidence from the above steps.\n"
        "If there are conflicts, explicitly explain how you resolve them.\n\n"
        "Use a layered, step-by-step reasoning chain: start from feature logic, then refine with statistics, then validate with similar cases.\n"
        "At the very end of your reasoning, on a separate line, output exactly:\n"
        "Prediction is <value>.\n"
        "where <value> is the predicted median house value in US Dollars as a numeric value.\n"
        "Do not add any other text after this line.\n\n"
    )

    INSTRUCTION_NORMAL_FINETUNE = (
        "You are a house price estimator. Based on the following housing and geographic features in California, explain why THE GIVEN PREDICTION LABEL for this area's median house value in US Dollars is plausible.\n\n"
        "At the very end of your reasoning, on a separate line, output exactly:\n"
        "Prediction is <value>.\n"
        "where <value> is the predicted median house value in US Dollars as a numeric value.\n"
        "Do not add any other text after this line.\n\n"
    )

    # Feature descriptions for better prompts (matching california.csv)
    FEATURE_DESCRIPTIONS = {
        "latitude": "Latitude",
        "longitude": "Longitude",
        "median_income": "Median income for households within a block of houses (measured in tens of thousands of US Dollars)",
        "total_rooms": "Total number of rooms within a block",
        "ocean_proximity": "Location of the house w.r.t ocean/sea.",
        "total_bedrooms": "Total number of bedrooms within a block",
        "population": "Total number of people residing within the block",
        "households": "Total number of households, a group of people residing within a home unit, for a block",
        "housing_median_age": "Median age of houses in the block",
        # Target description kept for reference; not shown in feature explanations.
        "median_house_value": "Median house value for households within a block (measured in US Dollars)",
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
            dataset_name="california",
            split=split,
            seed=seed,
            split_path=split_path,
            transform=transform,
        )


    # ------------------------------------------------------------------
    # Prompt building
    # ------------------------------------------------------------------
    def _build_feature_explanations(self) -> str:
        """Build feature explanations text from FEATURE_DESCRIPTIONS."""
        explanations = []
        for col, desc in self.FEATURE_DESCRIPTIONS.items():
            if col in self.feature_cols:
                explanations.append(f"- {col}: {desc}")
        if explanations:
            return "Feature explanations & Feature importance (high to low):\n" + "\n".join(explanations)
        return ""
    
    def build_prompt(
        self,
        idx: int,
        task: Literal["regression"] = "regression",
    ) -> dict:
        """
        Build an Alpaca-style prompt for the given sample index.

        Args:
            idx: Index of the sample in the current split.
            task: 'regression' to predict numeric value, or

        Returns:
            dict with keys: instruction, input, output
        """
        sample = self[idx]
        features = sample["features"]
        target = sample["target"]
        assert task == "regression"

        # Build input text from features using original column names
        lines = []
        for col in self.feature_cols:
            value = features[col]

            if value == "missing" or (isinstance(value, float) and str(value) == "nan"):
                val_str = "missing"
            else:
                val_str = str(value)

            lines.append(f"{col} is {val_str}")

        input_text = ". ".join(lines)

        # Build output depending on task

        # Clasification was deprecated, so we only support regression now.
        # Regression: use numeric target directly
        base_instruction = self.INSTRUCTION_REGRESSION
        base_instruction_reasoning = self.INSTRUCTION_REASONING_REGRESSION
        # Keep numeric but serialize as string for JSON corpus
        output = f"{float(target):.4f}"

        # Append feature explanations to instruction
        feature_explanations = self._build_feature_explanations()
        if feature_explanations:
            instruction = base_instruction.rstrip() + "\n\n" + feature_explanations + "\n"
            instruction_reasoning = (
                base_instruction_reasoning.rstrip() + "\n\n" + feature_explanations + "\n"
            )
        else:
            instruction = base_instruction
            instruction_reasoning = base_instruction_reasoning

        return {
            "instruction": instruction,
            "instruction_reasoning": instruction_reasoning,
            "input": input_text,
            "output": output,
        }

    def build_all_prompts(
        self,
        task: Literal["regression"] = "regression",
    ) -> list:
        """
        Build prompts for all samples in the current split.

        Args:
            task: 'regression'

        Returns:
            List of prompt dicts
        """
        return [self.build_prompt(i, task=task) for i in range(len(self))]

    # ------------------------------------------------------------------
    # Output parsing utilities
    # ------------------------------------------------------------------
    @staticmethod
    def parse_output(
        raw_text: str,
        task: Literal["regression"] = "regression",
    ) -> Optional[Union[float, str]]:
        """
        Parse model output for the California dataset.

        Args:
            raw_text: Raw model output text.
            task: 'regression' to return a numeric value (float),
        Returns:
            For regression: float value or None if cannot parse.
        """
        text = raw_text.strip().lower()

        # Regression mode: parse numeric value (tolerate explanations/currency).
        # Priority: look for an explicit prediction/value/estimate prefix;
        # otherwise fall back to the first numeric token in the text.
        return CaliforniaDataset._parse_float_from_text(text, prefer_prefixed=True)

    @staticmethod
    def _parse_float_from_text(text: str, prefer_prefixed: bool = False) -> Optional[float]:
        """
        Extract a float value from explicit prediction-like prefixes only.
        This intentionally avoids fallback parsing from arbitrary numbers
        in reasoning text (e.g., section indices like "(1)").
        """
        # Remove currency symbols for robustness
        cleaned = text.replace("$", "")

        if not prefer_prefixed:
            return None

        line_prefixed_matches = re.findall(
            r"^\s*\**\s*(prediction|value|estimate|output)\s*(?:is\s*)?[:=\-]?\s*\**\s*([-+]?\d[\d,]*\.?\d*(e[-+]?\d+)?)",
            cleaned,
            flags=re.IGNORECASE | re.MULTILINE,
        )
        if line_prefixed_matches:
            token = line_prefixed_matches[-1][1].replace(",", "")
            try:
                return float(token)
            except ValueError:
                return None

        any_prefixed_matches = re.findall(
            r"\**\s*(prediction|value|estimate|output)\s*(?:is\s*)?[:=\-]?\s*\**\s*([-+]?\d[\d,]*\.?\d*(e[-+]?\d+)?)",
            cleaned,
            flags=re.IGNORECASE,
        )
        if not any_prefixed_matches:
            return None

        token = any_prefixed_matches[-1][1].replace(",", "")
        try:
            return float(token)
        except ValueError:
            return None

    @staticmethod
    def output_to_target(
        output: Union[float, str, None],
        task: Literal["regression"] = "regression",
    ) -> Optional[Union[float, int]]:
        """
        Convert parsed output to numeric target.

        Args:
            output: Parsed output from `parse_output`.
            task: 'regression'.

        Returns:
            For regression: float value.
            None if conversion is not possible.
        """
        if output is None:
            return None

        # Regression: ensure float
        if isinstance(output, (int, float, np.floating)):
            return float(output)

        if isinstance(output, str):
            try:
                return float(output)
            except ValueError:
                return None

        return None


# Convenience function
def get_california_dataset(
    split: Optional[Literal["train", "valid", "test"]] = None,
    seed: int = 42,
) -> CaliforniaDataset:
    """Get California dataset with specified split."""
    return CaliforniaDataset(split=split, seed=seed)


def convert_target_to_text(
    y,
    sample: Optional[dict] = None,
    metadata: Optional[dict] = None,
    task: Optional[str] = None,
) -> str:
    """
    Convert target value to a natural-language sentence describing the ground-truth label.
    
    This function is used by comparison.py for RAG retrieval to generate
    human-readable label text for each retrieved training example.
    
    Args:
        y: Target value from the California dataset.
           Can be numeric (median house value) or string ('above_median'/'below_median').
        sample: Optional dictionary containing the sample's features.
                Not used for California dataset but available for datasets
                that need context.
        metadata: Optional dictionary containing additional metadata.
                  Can include 'median' key for custom threshold.
        task: Optional task type ('regression' or 'classification').
              If 'regression', returns the actual numeric value.
              Defaults to 'regression'.
    
    Returns:
        A natural-language sentence describing the target label.
    
    Examples:
        >>> convert_target_to_text(250000.0, task='regression')
        'The median house value for this area is $250000.'
    """
    
    # Handle numeric values
    try:
        numeric_val = float(y)
        return f"The median house value for this area is ${numeric_val:.0f}."
    except (ValueError, TypeError):
        return f"The house value is: {y}"
