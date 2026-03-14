import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from app.mlflow_utils import log_artifact_if_exists
from app.utils import ensure_dir


def save_training_curves(history: dict, settings) -> str:
    ensure_dir(settings.plots_dir)
    output_path = os.path.join(settings.plots_dir, "training_curves.png")

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    axes[0].plot(history["train_acc"], label="Train", marker="o")
    axes[0].plot(history["val_acc"], label="Val", marker="s")
    axes[0].set_title("Accuracy")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Accuracy")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(history["train_loss"], label="Train", marker="o")
    axes[1].plot(history["val_loss"], label="Val", marker="s")
    axes[1].set_title("Loss")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Loss")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()

    return output_path


def save_confusion_matrix_plot(confusion_matrix, classes, settings) -> str:
    ensure_dir(settings.plots_dir)
    output_path = os.path.join(settings.plots_dir, "confusion_matrix.png")

    plt.figure(figsize=(10, 8))
    sns.heatmap(
        confusion_matrix,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=classes,
        yticklabels=classes,
        cbar_kws={"label": "Count"},
        annot_kws={"size": 14, "weight": "bold"},
    )
    plt.title("Confusion Matrix", fontsize=16, fontweight="bold", pad=20)
    plt.ylabel("True Label")
    plt.xlabel("Predicted Label")
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()

    return output_path


def save_error_visualization(
    df_test,
    y_true,
    y_pred,
    y_pred_proba,
    errors_idx,
    classes,
    settings,
) -> str | None:
    if len(errors_idx) == 0:
        return None

    ensure_dir(settings.plots_dir)
    output_path = os.path.join(settings.plots_dir, "errors_visualization.png")

    n_errors_to_show = min(6, len(errors_idx))
    sample_errors = np.random.choice(errors_idx, n_errors_to_show, replace=False)

    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    axes = axes.flatten()

    for i, idx in enumerate(sample_errors):
        img = np.array(df_test.iloc[idx]["image_array"], dtype=np.float32)

        if img.ndim == 3 and img.shape[-1] > 1:
            display_img = img[:, :, 0]
        elif img.ndim == 3 and img.shape[-1] == 1:
            display_img = img[:, :, 0]
        else:
            display_img = img

        true_label = classes[y_true[idx]]
        pred_label = classes[y_pred[idx]]
        confidence = y_pred_proba[idx][y_pred[idx]] * 100

        axes[i].imshow(display_img, cmap="gray")
        axes[i].set_title(
            f"True: {true_label}\nPred: {pred_label} ({confidence:.1f}%)",
            fontsize=10,
            color="red",
        )
        axes[i].axis("off")

    for j in range(i + 1, len(axes)):
        axes[j].axis("off")

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()

    return output_path