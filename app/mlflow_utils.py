import os

import mlflow
import mlflow.pytorch


def setup_mlflow(tracking_uri: str, experiment_name: str) -> None:
    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment(experiment_name)

    if mlflow.active_run() is not None:
        mlflow.end_run()


def log_training_params(
    model_type: str,
    num_classes: int,
    epochs: int,
    batch_size: int,
    learning_rate: float,
    dataset_name: str,
    img_size: int,
) -> None:
    mlflow.log_param("model_type", model_type)
    mlflow.log_param("num_classes", num_classes)
    mlflow.log_param("epochs", epochs)
    mlflow.log_param("batch_size", batch_size)
    mlflow.log_param("learning_rate", learning_rate)
    mlflow.log_param("dataset", dataset_name)
    mlflow.log_param("img_size", img_size)


def log_epoch_metrics(
    epoch: int,
    train_loss: float,
    train_acc: float,
    val_loss: float,
    val_acc: float,
) -> None:
    mlflow.log_metric("train_loss", train_loss, step=epoch)
    mlflow.log_metric("train_accuracy", train_acc, step=epoch)
    mlflow.log_metric("val_loss", val_loss, step=epoch)
    mlflow.log_metric("val_accuracy", val_acc, step=epoch)


def log_test_metrics(
    accuracy: float,
    precision: float,
    recall: float,
    f1: float,
) -> None:
    mlflow.log_metric("test_accuracy", float(accuracy))
    mlflow.log_metric("test_precision", float(precision))
    mlflow.log_metric("test_recall", float(recall))
    mlflow.log_metric("test_f1", float(f1))


def log_artifact_if_exists(filepath: str, artifact_path: str | None = None) -> None:
    if os.path.exists(filepath):
        mlflow.log_artifact(filepath, artifact_path=artifact_path)


def log_pytorch_model(model) -> None:
    model.eval()
    mlflow.pytorch.log_model(
        pytorch_model=model,
        name="model",
    )