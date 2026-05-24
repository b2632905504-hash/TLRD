"""
Dataset class registry and helpers.

Keeps a mapping from dataset name to its corresponding TabularDataset subclass
so the corpus builder can instantiate datasets without hard-coding imports.
"""

import sys
from pathlib import Path
from typing import Dict, Type, Optional

from data.dataset import TabularDataset

# Ensure repository-local imports work when running as a script
_SCRIPT_DIR = Path(__file__).resolve().parent
if str(_SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPT_DIR))
_PARENT_DIR = _SCRIPT_DIR.parent
if str(_PARENT_DIR) not in sys.path:
    sys.path.insert(0, str(_PARENT_DIR))


DATASET_CLASS_REGISTRY: Dict[str, Type[TabularDataset]] = {}


def register_dataset_class(name: str, cls: Type[TabularDataset]) -> None:
    """Register a dataset class for corpus building."""
    DATASET_CLASS_REGISTRY[name] = cls


def get_dataset_class(name: str) -> Type[TabularDataset]:
    """Return the dataset class for a given name."""
    if name in DATASET_CLASS_REGISTRY:
        return DATASET_CLASS_REGISTRY[name]
    raise ValueError(f"Dataset class for {name} not found")


def _import_dataset(module_paths: list[str], class_name: str) -> Optional[Type[TabularDataset]]:
    """
    Try importing the given class from multiple module paths.

    This avoids hard failures when the package is invoked from different working
    directories (e.g., with or without `data.` prefix).
    """
    last_error: Optional[Exception] = None
    for module_path in module_paths:
        try:
            module = __import__(module_path, fromlist=[class_name])
            return getattr(module, class_name)
        except ImportError as e:
            last_error = e
            continue
    if last_error:
        print(f"Warning: Could not import {class_name}: {last_error}")
    return None


def register_known_datasets() -> None:
    """Register built-in datasets shipped with this repository."""
    adult_cls = _import_dataset(
        ["adult.adult_dataset", "data.adult.adult_dataset"],
        "AdultDataset",
    )
    if adult_cls:
        register_dataset_class("adult", adult_cls)

    california_cls = _import_dataset(
        ["california.california_dataset", "data.california.california_dataset"],
        "CaliforniaDataset",
    )
    if california_cls:
        register_dataset_class("california", california_cls)

    diamonds_cls = _import_dataset(
        ["diamonds.diamonds_dataset", "data.diamonds.diamonds_dataset"],
        "DiamondsDataset",
    )
    if diamonds_cls:
        register_dataset_class("diamonds", diamonds_cls)

    home_credit_cls = _import_dataset(
        ["home_credit.home_credit_dataset", "data.home_credit.home_credit_dataset"],
        "HomeCreditDataset",
    )
    if home_credit_cls:
        register_dataset_class("home_credit", home_credit_cls)

    okcupid_cls = _import_dataset(
        ["okcupid_stem.okcupid_stem_dataset", "data.okcupid_stem.okcupid_stem_dataset"],
        "OkCupidStemDataset",
    )
    if okcupid_cls:
        register_dataset_class("okcupid_stem", okcupid_cls)

    diabetes_130us_cls = _import_dataset(
        [
            "diabetes_130US.diabetes_130US_dataset",
            "data.diabetes_130US.diabetes_130US_dataset",
        ],
        "Diabetes130USDataset",
    )
    if diabetes_130us_cls:
        register_dataset_class("diabetes_130US", diabetes_130us_cls)


# Eagerly register built-in datasets on import
register_known_datasets()
