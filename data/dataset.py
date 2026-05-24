"""
Base Dataset Classes for Tabular Data
Supports loading pre-computed splits from split_{seed}.pth files
"""

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Any, Optional, List, Literal, Tuple

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset


TaskType = Literal["regression", "binary", "multiclass"]


@dataclass
class DatasetConfig:
    """Configuration for a dataset"""
    name: str
    csv_path: str
    target_col: str
    task_type: TaskType
    # Optional human-readable target name for prompt/stats display.
    target_display_name: Optional[str] = None
    cat_cols: Optional[List[str]] = None
    num_cols: Optional[List[str]] = None  # If None, inferred as non-cat columns
    target_label_display_map: Optional[Dict[str, str]] = None


# ==================== Dataset Registry ====================

# Resolve repository-local data directory (data/ alongside this file)
DATA_DIR = Path(__file__).resolve().parent

DATASET_REGISTRY: Dict[str, DatasetConfig] = {
    "adult": DatasetConfig(
        name="adult",
        csv_path=str(DATA_DIR / "adult" / "adult.csv"),
        target_col="income",
        task_type="binary",
        cat_cols=[
            "marital-status",
            "occupation",
            "gender",
            "relationship",
            "workclass",
            "race",
            "native-country",
            "education",
        ],
        num_cols=[
            "age",
            "capital-gain",
            "educational-num",
            "hours-per-week",
            "capital-loss",
            "fnlwgt",
        ],
    ),
    "california": DatasetConfig(
        name="california",
        csv_path=str(DATA_DIR / "california" / "california.csv"),
        target_col="median_house_value",
        task_type="regression",
        cat_cols=["ocean_proximity"],
        num_cols=[
            "latitude",
            "longitude",
            "median_income",
            "total_rooms",
            "total_bedrooms",
            "population",
            "households",
            "housing_median_age",
        ],
    ),
    "diamonds": DatasetConfig(
        name="diamonds",
        csv_path=str(DATA_DIR / "diamonds" / "diamonds.csv"),
        target_col="price",
        task_type="regression",
        cat_cols=["clarity", "color", "cut"],
        # Keep explicit order aligned with feature-importance ranking.
        num_cols=["carat", "y", "x", "z", "depth", "table"],
    ),
    "home_credit": DatasetConfig(
        name="home_credit",
        csv_path=str(DATA_DIR / "home_credit" / "application_train.csv"),
        target_col="TARGET",
        task_type="binary",
        target_label_display_map={
            "0": "non_default",
            "1": "default",
        },
        cat_cols=[
            "CODE_GENDER",
            "NAME_EDUCATION_TYPE",
            "OCCUPATION_TYPE",
            "FLAG_OWN_CAR",
            "ORGANIZATION_TYPE",
            "FLAG_DOCUMENT_3",
        ],
        num_cols=[
            "EXT_SOURCE_3",
            "EXT_SOURCE_2",
            "AMT_GOODS_PRICE",
            "EXT_SOURCE_1",
            "AMT_CREDIT",
            "DAYS_EMPLOYED",
            "DAYS_BIRTH",
            "AMT_ANNUITY",
            "DAYS_ID_PUBLISH",
        ],
    ),
    "okcupid_stem": DatasetConfig(
        name="okcupid_stem",
        csv_path=str(DATA_DIR / "okcupid_stem" / "okcupid_stem.csv"),
        target_col="job",
        task_type="multiclass",
        cat_cols=[
            "education",
            "sex",
            "location",
            "sign",
            "religion",
            "speaks",
            "orientation",
            "body_type",
            "ethnicity",
            "drinks",
            "offspring",
            "smokes",
            "pets",
            "status",
            "diet",
            "drugs",
        ],
        num_cols=["age", "income", "height"],
    ),
    "diabetes_130US": DatasetConfig(
        name="diabetes_130US",
        csv_path=str(DATA_DIR / "diabetes_130US" / "diabetic_data_paper_aligned_3class.csv"),
        target_col="readmitted",
        task_type="multiclass",
        cat_cols=[
            "discharge_disposition_id",
            "diag_1",
            "age",
            "diabetesMed",
            "medical_specialty",
            "admission_source_id",
            "admission_type_id",
            "diag_2",
        ],
        num_cols=[
            "number_inpatient",
            "number_diagnoses",
            "time_in_hospital",
            "num_lab_procedures",
            "num_medications",
            "number_outpatient",
            "num_procedures",
        ],
    ),
}

def register_dataset(name: str, config: DatasetConfig):
    """Register a new dataset configuration"""
    DATASET_REGISTRY[name] = config


def get_dataset_config(name: str) -> DatasetConfig:
    """Get dataset configuration by name"""
    if name not in DATASET_REGISTRY:
        raise ValueError(f"Dataset '{name}' not found. Available: {list(DATASET_REGISTRY.keys())}")
    return DATASET_REGISTRY[name]


# ==================== Split Loading ====================

def load_split(split_path: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Load split indices from a .pth file"""
    data = torch.load(split_path, weights_only=False)
    idx_train = data["idx_train"].numpy() if isinstance(data["idx_train"], torch.Tensor) else data["idx_train"]
    idx_valid = data["idx_valid"].numpy() if isinstance(data["idx_valid"], torch.Tensor) else data["idx_valid"]
    idx_test = data["idx_test"].numpy() if isinstance(data["idx_test"], torch.Tensor) else data["idx_test"]
    return idx_train, idx_valid, idx_test


def get_default_split_path(config: DatasetConfig, seed: int) -> str:
    """Get the default split file path for a dataset"""
    data_dir = Path(config.csv_path).parent
    return str(data_dir / f"split_{seed}.pth")


# ==================== Base Dataset Class ====================

class TabularDataset(Dataset):
    """
    Base PyTorch Dataset for tabular data with split support.
    
    Usage:
        # Load training split
        train_ds = TabularDataset("adult", split="train", seed=42)
        
        # Load validation split  
        val_ds = TabularDataset("adult", split="valid", seed=42)
        
        # Load test split
        test_ds = TabularDataset("adult", split="test", seed=42)
        
        # Load full dataset (no split)
        full_ds = TabularDataset("adult", split=None)
    """

    def __init__(
        self,
        dataset_name: str,
        split: Optional[Literal["train", "valid", "test"]] = None,
        seed: int = 42,
        split_path: Optional[str] = None,
        transform: Optional[callable] = None,
    ):
        """
        Args:
            dataset_name: Name of the registered dataset
            split: Which split to load ('train', 'valid', 'test', or None for full)
            seed: Random seed used for split file (split_{seed}.pth)
            split_path: Custom path to split file (overrides default)
            transform: Optional transform to apply to samples
        """
        self.config = get_dataset_config(dataset_name)
        self.split = split
        self.seed = seed
        self.transform = transform

        # Load full dataframe
        self.df_full = pd.read_csv(self.config.csv_path)

        # Infer column types
        self._infer_columns()

        # Load split if specified
        if split is not None:
            if split_path is None:
                split_path = get_default_split_path(self.config, seed)

            if not Path(split_path).exists():
                raise FileNotFoundError(
                    f"Split file not found: {split_path}\n"
                    f"Please run preprocess.py first: python preprocess.py --seed {seed}"
                )

            idx_train, idx_valid, idx_test = load_split(split_path)

            if split == "train":
                self.indices = idx_train
            elif split == "valid":
                self.indices = idx_valid
            elif split == "test":
                self.indices = idx_test
            else:
                raise ValueError(f"Invalid split: {split}")

            self.df = self.df_full.iloc[self.indices].reset_index(drop=True)
        else:
            self.indices = np.arange(len(self.df_full))
            self.df = self.df_full

    def _infer_columns(self):
        """Infer categorical and numerical columns"""
        all_cols = [c for c in self.df_full.columns if c != self.config.target_col]

        # Categorical columns
        if self.config.cat_cols is not None:
            self.cat_cols = [c for c in self.config.cat_cols if c in all_cols]
        else:
            self.cat_cols = [
                c for c in all_cols
                if self.df_full[c].dtype == "object" or str(self.df_full[c].dtype).startswith("category")
            ]

        # Numerical columns
        if self.config.num_cols is not None:
            self.num_cols = [c for c in self.config.num_cols if c in all_cols]
        else:
            self.num_cols = [c for c in all_cols if c not in self.cat_cols]

        self.feature_cols = self.num_cols + self.cat_cols

    @property
    def target_col(self) -> str:
        return self.config.target_col

    @property
    def task_type(self) -> TaskType:
        return self.config.task_type

    @property
    def n_features(self) -> int:
        return len(self.feature_cols)

    @property
    def n_classes(self) -> int:
        """Number of classes for classification tasks"""
        if self.config.task_type == "regression":
            return 0
        return len(self.df_full[self.config.target_col].unique())

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        """Get a single sample as a dictionary"""
        row = self.df.iloc[idx]

        # Extract features
        features = {}
        for col in self.feature_cols:
            value = row[col]
            if pd.isna(value):
                features[col] = "missing" if col in self.cat_cols else np.nan
            elif col in self.cat_cols and value == "?":
                features[col] = "missing"
            else:
                features[col] = value

        # Extract target
        target = row[self.config.target_col]

        sample = {
            "features": features,
            "target": target,
            "index": idx,
            "original_index": self.indices[idx] if self.split else idx,
        }

        if self.transform:
            sample = self.transform(sample)

        return sample

    def get_features_array(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Get features as numpy arrays (numerical and categorical separately).
        Returns: (X_num, X_cat) where X_cat is label-encoded
        """
        X_num = self.df[self.num_cols].values.astype(np.float32) if self.num_cols else None
        
        if self.cat_cols:
            X_cat = self.df[self.cat_cols].copy()
            # Handle missing values and convert to string
            for col in self.cat_cols:
                X_cat[col] = X_cat[col].fillna("missing").astype(str)
            X_cat = X_cat.values
        else:
            X_cat = None

        return X_num, X_cat

    def get_targets(self) -> np.ndarray:
        """Get targets as numpy array"""
        return self.df[self.config.target_col].values

    def get_dataframe(self) -> pd.DataFrame:
        """Get the underlying dataframe"""
        return self.df.copy()

    def get_class_distribution(self) -> Dict[Any, int]:
        """Get class distribution for classification tasks"""
        return self.df[self.config.target_col].value_counts().to_dict()

    def __repr__(self) -> str:
        return (
            f"TabularDataset(name={self.config.name}, split={self.split}, "
            f"n_samples={len(self)}, n_features={self.n_features}, "
            f"task={self.config.task_type})"
        )
    
    def build_prompt(self, idx: int) -> str:
        """Build prompt for LLM - to be implemented by subclasses"""
        raise NotImplementedError

    def parse_output(self, raw_text: str) -> Optional[str]:
        """Parse output from model response"""
        raise NotImplementedError
