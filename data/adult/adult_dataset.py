"""
Adult Dataset Class with LLM Prompt Building Support
"""

import re
from typing import Optional, Literal

from data.dataset import TabularDataset, DatasetConfig, DATASET_REGISTRY


class AdultDataset(TabularDataset):
    """
    Adult Income Dataset for binary classification.
    
    Task: Predict whether income exceeds $50K/year based on census data.
    Target: income (<=50K or >50K)
    
    Usage:
        # Load training split
        train_ds = AdultDataset(split="train", seed=42)
        
        # Build prompt for a sample
        prompt = train_ds.build_prompt(0)
    """
    
    # Class-level instruction template
    INSTRUCTION = (
        "You are an income classifier. "
        "Based on the following census features from 1990s below, predict whether this person's annual income exceeds $50K. "
        "Please explain your reasoning.\n\n"
        "At the very end of your reasoning, on a separate line, output exactly:\n" 
        "Prediction is <value>.\n"
        "where <value> is either yes or no.\n"
        "yes means income > $50K; no means income <= $50K.\n"
        "Do not add any other text after this line.\n\n"
    )
 

    INSTRUCTION_REASONING = (
        "You are an income classifier. Based on the following census features from the 1990s, conduct in-depth qualitative analysis to predict whether this person's annual income exceeds $50K. "
        "You should use dataset statistics and similar historical cases to assist your reasoning. Please explain your reasoning in the following way.\n\n"
        "Reasoning format:\n"
        "(1). Self feature-based reasoning:\n"
        "(2). Global statistics-based reasoning:\n"
        "(3). Similar-cases reasoning:\n"
        "(4). Final Resolution:\n"
        "At the very end of your reasoning, on a separate line, output exactly:\n"
        "Prediction is <value>.\n"
        "where <value> is either yes or no.\n"
        "yes means income > $50K; no means income <= $50K.\n"
        "Do not add any other text after this line.\n\n"
    )

    INSTRUCTION_FINETUNE = (
        "You are an income classifier. Based on the following census features from the 1990s, conduct in-depth qualitative analysis to explain why THE GIVEN PREDICTION LABEL for this person's income is plausible (yes means income > $50K; no means income <= $50K). You are also provided with dataset statistics and similar historical cases to assist your reasoning. Please explain your reasoning in the following way.\n\n"
        "Reasoning format:\n"

        "(1). Self feature-based reasoning:\n"
        "- Analyze each feature from \"Current Sample Features\" strictly using general economic logic and 1990s U.S. labor market common sense.\n"
        "Execution Rules:\n"
        "- Briefly explain your intuition, and state its directional impact on income.\n"
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
        "yes means income > $50K; no means income <= $50K.\n"
        "Do not add any other text after this line.\n\n"

    )

    INSTRUCTION_NORMAL_FINETUNE = (
        "You are an income classifier. Based on the following census features from the 1990s, explain why THE GIVEN PREDICTION LABEL for this person's income is plausible (yes means income > $50K; no means income <= $50K).\n\n"
        "At the very end of your reasoning, on a separate line, output exactly:\n"
        "Prediction is <value>.\n"
        "where <value> is either yes or no.\n"
        "yes means income > $50K; no means income <= $50K.\n"
        "Do not add any other text after this line.\n\n"
    )


    
    
    # Feature descriptions for better prompts
    FEATURE_DESCRIPTIONS = {
        "marital-status": "Marital status",
        "age": "Age",
        "capital-gain": "Capital gain",
        "educational-num": "Education level (numeric)",
        "occupation": "Occupation",
        "hours-per-week": "Hours worked per week",
        "capital-loss": "Capital loss",
        "gender": "Gender",
        "relationship": "Relationship status",
        "fnlwgt": "Final weight (census sampling weight)",
        "workclass": "Work class",
        "race": "Race",
        "native-country": "Native country",
        "education": "Highest education level",
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
            dataset_name="adult",
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
            elif value == "?":
                val_str = "missing"
            else:
                val_str = str(value)
            
            lines.append(f"{col} is {val_str}")
        
        input_text = ". ".join(lines)
        
        # Convert target to yes/no
        # >50K means income exceeds 50K -> yes
        # <=50K means income does not exceed 50K -> no
        answer = "yes" if target == ">50K" else "no"
        
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
            #"instruction_reasoning": instruction_reasoning,
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
        Parse yes/no from model output for Adult dataset.
        
        Args:
            raw_text: Raw model output text
            
        Returns:
            'yes', 'no', or None if cannot parse
        """
        text = raw_text.strip().lower()
        
        # Priority 1: Line-start "Prediction is <yes/no>."
        line_pred_matches = re.findall(
            r"^\s*\**\s*prediction\s+is\s+\**\s*(yes|no)\b",
            text,
            flags=re.IGNORECASE | re.MULTILINE,
        )
        if line_pred_matches:
            return line_pred_matches[-1].lower()

        # Priority 2: Anywhere "Prediction is <yes/no>"
        any_pred_matches = re.findall(
            r"\**\s*prediction\s+is\s+\**\s*(yes|no)\b",
            text,
            flags=re.IGNORECASE,
        )
        if any_pred_matches:
            return any_pred_matches[-1].lower()
        
        return None
    
    @staticmethod
    def output_to_target(output: str) -> int:
        """
        Convert output label to binary target.
        
        Args:
            output: 'yes' or 'no'
            
        Returns:
            1 for yes (income > 50K), 0 for no (income <= 50K)
        """
        return 1 if output and output.lower() == "yes" else 0
    
    @staticmethod
    def target_to_output(target) -> str:
        """
        Convert binary target to output label.
        
        Args:
            target: 1 or 0, or '>50K' / '<=50K'
            
        Returns:
            'yes' or 'no'
        """
        if isinstance(target, str):
            return "yes" if target == ">50K" else "no"
        return "yes" if target == 1 else "no"


# Convenience function
def get_adult_dataset(
    split: Optional[Literal["train", "valid", "test"]] = None,
    seed: int = 42,
) -> AdultDataset:
    """Get Adult dataset with specified split"""
    return AdultDataset(split=split, seed=seed)


def convert_target_to_text(y, sample: Optional[dict] = None, metadata: Optional[dict] = None) -> str:
    """
    Convert target value to a natural-language sentence describing the ground-truth label.
    
    This function is used by comparison.py for RAG retrieval to generate
    human-readable label text for each retrieved training example.
    
    Args:
        y: Target value from the Adult dataset.
           Can be '>50K', '<=50K', 1, 0, 'yes', 'no'
        sample: Optional dictionary containing the sample's features.
                Not used for Adult dataset but available for datasets
                that need context.
        metadata: Optional dictionary containing additional metadata.
                  Not used for Adult dataset but available for extensibility.
    
    Returns:
        A natural-language sentence describing the target label.
    
    Examples:
        >>> convert_target_to_text('>50K')
        'This person has income > $50K.'
        >>> convert_target_to_text('<=50K')
        'This person has income ≤ $50K.'
        >>> convert_target_to_text(1)
        'This person has income > $50K.'
        >>> convert_target_to_text('yes')
        'This person has income > $50K.'
    """
    # Determine if positive class (income > $50K)
    is_positive = False
    if isinstance(y, str):
        y_lower = y.lower().strip()
        if y_lower in ('>50k', 'yes', '1', 'true'):
            is_positive = True
    elif isinstance(y, (int, float)):
        is_positive = y > 0
    
    if is_positive:
        return "This person has income > $50K."
    else:
        return "This person has income ≤ $50K."


if __name__ == "__main__":
    # Quick test
    ds = AdultDataset(split="train", seed=42)
    print(ds)
    print(f"\nClass distribution: {ds.get_class_distribution()}")
    print(f"\nSample prompt:\n")
    prompt = ds.build_prompt(0)
    print(f"Instruction: {prompt['instruction'][:100]}...")
    print(f"Input: {prompt['input'][:200]}...")
    print(f"Output: {prompt['output']}")
