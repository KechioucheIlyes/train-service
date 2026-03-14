from dataclasses import dataclass
import os
from dotenv import load_dotenv

load_dotenv()


def _get_env(name: str, default: str | None = None) -> str:
    value = os.getenv(name, default)
    if value is None:
        raise ValueError(f"Missing required environment variable: {name}")
    return value


@dataclass
class Settings:
    kaggle_username: str
    kaggle_key: str

    mlflow_tracking_uri: str
    mlflow_experiment_name: str

    dataset_slug: str

    data_dir: str
    preprocessed_dir: str
    output_dir: str
    model_dir: str
    plots_dir: str

    run_name: str

    batch_size: int
    epochs: int
    learning_rate: float
    img_size: int
    early_stopping_patience: int
    lr_scheduler_patience: int
    lr_scheduler_factor: float
    min_lr: float
    random_state: int

    registry_api_url: str
    registry_api_token: str


def get_settings() -> Settings:
    return Settings(
        kaggle_username=_get_env("KAGGLE_USERNAME"),
        kaggle_key=_get_env("KAGGLE_KEY"),
        mlflow_tracking_uri=_get_env("MLFLOW_TRACKING_URI"),
        mlflow_experiment_name=_get_env("MLFLOW_EXPERIMENT_NAME", "Shifaa_Finetuning"),
        dataset_slug=_get_env("DATASET_SLUG"),
        data_dir=_get_env("DATA_DIR", "./data"),
        preprocessed_dir=_get_env("PREPROCESSED_DIR", "./data/preprocessed"),
        output_dir=_get_env("OUTPUT_DIR", "./outputs"),
        model_dir=_get_env("MODEL_DIR", "./outputs/models"),
        plots_dir=_get_env("PLOTS_DIR", "./outputs/plots"),
        run_name=_get_env("RUN_NAME", "shifaa_s8_run"),
        batch_size=int(_get_env("BATCH_SIZE", "32")),
        epochs=int(_get_env("EPOCHS", "20")),
        learning_rate=float(_get_env("LEARNING_RATE", "0.0001")),
        img_size=int(_get_env("IMG_SIZE", "224")),
        early_stopping_patience=int(_get_env("EARLY_STOPPING_PATIENCE", "7")),
        lr_scheduler_patience=int(_get_env("LR_SCHEDULER_PATIENCE", "3")),
        lr_scheduler_factor=float(_get_env("LR_SCHEDULER_FACTOR", "0.5")),
        min_lr=float(_get_env("MIN_LR", "0.0000001")),
        random_state=int(_get_env("RANDOM_STATE", "42")),
        registry_api_url=_get_env("REGISTRY_API_URL", "http://localhost:8082"),
        registry_api_token=_get_env("REGISTRY_API_TOKEN", "change_me"),
    )