import json
import os

import numpy as np
import torch
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)

from app.mlflow_utils import log_test_metrics, log_artifact_if_exists


def run_evaluation_pipeline(
    model,
    test_loader,
    df_test,
    label_encoder,
    device,
    settings,
):
    print("\n" + "=" * 70)
    print("ÉVALUATION SUR LE TEST SET")
    print("=" * 70)

    model.eval()

    all_preds = []
    all_labels = []
    all_probs = []

    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)
            probs = torch.softmax(outputs, dim=1)
            _, preds = torch.max(outputs, 1)

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())

    y_pred = np.array(all_preds)
    y_true = np.array(all_labels)
    y_pred_proba = np.array(all_probs)

    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, average="weighted", zero_division=0)
    recall = recall_score(y_true, y_pred, average="weighted", zero_division=0)
    f1 = f1_score(y_true, y_pred, average="weighted", zero_division=0)

    print(f"\n📊 MÉTRIQUES GLOBALES :")
    print(f"   Accuracy  : {accuracy*100:.2f}%")
    print(f"   Precision : {precision*100:.2f}%")
    print(f"   Recall    : {recall*100:.2f}%")
    print(f"   F1-Score  : {f1*100:.2f}%")

    report = classification_report(
        y_true,
        y_pred,
        target_names=label_encoder.classes_,
        digits=4,
        zero_division=0,
        output_dict=False,
    )

    print(f"\n📋 RAPPORT DE CLASSIFICATION PAR CLASSE :\n")
    print(report)

    cm = confusion_matrix(y_true, y_pred)

    errors_idx = np.where(y_pred != y_true)[0]
    error_analysis = {}

    for idx in errors_idx:
        true_class = label_encoder.classes_[y_true[idx]]
        pred_class = label_encoder.classes_[y_pred[idx]]
        error_type = f"{true_class} -> {pred_class}"
        error_analysis[error_type] = error_analysis.get(error_type, 0) + 1

    results = {
        "accuracy": float(accuracy),
        "precision": float(precision),
        "recall": float(recall),
        "f1_score": float(f1),
        "confusion_matrix": cm.tolist(),
        "classes": label_encoder.classes_.tolist(),
        "model_type": "Shifaa Fine-tuned",
        "framework": "PyTorch",
        "num_errors": int(len(errors_idx)),
        "error_analysis": error_analysis,
    }

    results_path = os.path.join(settings.model_dir, "results.json")
    with open(results_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    log_test_metrics(
        accuracy=accuracy,
        precision=precision,
        recall=recall,
        f1=f1,
    )

    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "confusion_matrix": cm,
        "y_true": y_true,
        "y_pred": y_pred,
        "y_pred_proba": y_pred_proba,
        "errors_idx": errors_idx,
        "error_analysis": error_analysis,
        "results_path": results_path,
    }