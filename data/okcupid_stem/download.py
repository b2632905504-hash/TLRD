#!/usr/bin/env python3
"""Download OpenML task data and save as CSV in this directory."""

from pathlib import Path

import openml
import pandas as pd


TASK_ID = 359993
OUT_PATH = Path(__file__).resolve().parent / "okcupid_stem.csv"


def main() -> None:
    task = openml.tasks.get_task(TASK_ID, download_data=True)
    dataset = task.get_dataset()

    target_name = task.target_name
    X, y, _, _ = dataset.get_data(dataset_format="dataframe", target=target_name)

    df = X.copy()
    if y is not None:
        y_series = pd.Series(y, name=target_name or "target")
        df[y_series.name] = y_series

    df.to_csv(OUT_PATH, index=False)
    print(f"task_id={TASK_ID}")
    print(f"dataset_id={dataset.dataset_id}")
    print(f"target={target_name}")
    print(f"saved={OUT_PATH}")
    print(f"shape={df.shape}")


if __name__ == "__main__":
    main()
