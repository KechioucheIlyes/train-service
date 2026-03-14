import json
import os
import shutil
import subprocess
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.utils.class_weight import compute_class_weight

from app.config import Settings
from app.utils import ensure_dirs


def setup_kaggle_credentials(settings: Settings) -> None:
    kaggle_dir = Path.home() / ".kaggle"
    kaggle_dir.mkdir(parents=True, exist_ok=True)

    kaggle_json_path = kaggle_dir / "kaggle.json"
    payload = {
        "username": settings.kaggle_username,
        "key": settings.kaggle_key,
    }

    with open(kaggle_json_path, "w", encoding="utf-8") as f:
        json.dump(payload, f)

    os.chmod(kaggle_json_path, 0o600)


def download_dataset(settings: Settings) -> str:
    ensure_dirs([settings.data_dir, settings.preprocessed_dir])

    archive_path = os.path.join(settings.data_dir, "dataset.zip")

    command = [
        "kaggle",
        "datasets",
        "download",
        "-d",
        settings.dataset_slug,
        "-p",
        settings.data_dir,
        "-o",
    ]
    subprocess.run(command, check=True)

    downloaded_files = list(Path(settings.data_dir).glob("*.zip"))
    if not downloaded_files:
        raise FileNotFoundError("No dataset zip file found after Kaggle download.")

    latest_zip = max(downloaded_files, key=lambda p: p.stat().st_mtime)

    if os.path.exists(settings.preprocessed_dir):
        shutil.rmtree(settings.preprocessed_dir)

    Path(settings.preprocessed_dir).mkdir(parents=True, exist_ok=True)

    unzip_command = [
        "unzip",
        "-q",
        str(latest_zip),
        "-d",
        settings.preprocessed_dir,
    ]
    subprocess.run(unzip_command, check=True)

    return str(latest_zip)


def load_dataframe(settings: Settings) -> tuple[pd.DataFrame, LabelEncoder]:
    pickle_path = os.path.join(settings.preprocessed_dir, "s8_df_reduced_224.pkl")
    if not os.path.exists(pickle_path):
        raise FileNotFoundError(f"Pickle file not found: {pickle_path}")

    df = pd.read_pickle(pickle_path)

    label_encoder = LabelEncoder()
    df["label_encoded"] = label_encoder.fit_transform(df["classe"])

    return df, label_encoder


def split_dataframe(
    df: pd.DataFrame,
    random_state: int,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    train_idx, temp_idx = train_test_split(
        df.index,
        test_size=0.3,
        stratify=df["label_encoded"],
        random_state=random_state,
    )

    val_idx, test_idx = train_test_split(
        temp_idx,
        test_size=0.5,
        stratify=df.loc[temp_idx, "label_encoded"],
        random_state=random_state,
    )

    df_train = df.loc[train_idx].reset_index(drop=True)
    df_val = df.loc[val_idx].reset_index(drop=True)
    df_test = df.loc[test_idx].reset_index(drop=True)

    return df_train, df_val, df_test


def compute_class_weights_dict(df_train: pd.DataFrame) -> dict[int, float]:
    class_weights_array = compute_class_weight(
        class_weight="balanced",
        classes=np.unique(df_train["label_encoded"]),
        y=df_train["label_encoded"],
    )
    return dict(enumerate(class_weights_array))


def prepare_data(settings: Settings):
    setup_kaggle_credentials(settings)
    download_dataset(settings)
    df, label_encoder = load_dataframe(settings)
    df_train, df_val, df_test = split_dataframe(df, settings.random_state)
    class_weights = compute_class_weights_dict(df_train)

    return {
        "df_full": df,
        "df_train": df_train,
        "df_val": df_val,
        "df_test": df_test,
        "label_encoder": label_encoder,
        "class_weights": class_weights,
    }