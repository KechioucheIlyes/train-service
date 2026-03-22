import mlflow

from app.config import get_settings
from app.data import prepare_data
from app.mlflow_utils import (
    log_artifact_if_exists,
    log_pytorch_model,
    setup_mlflow,
)
from app.model import (
    build_training_components,
    configure_model_for_finetuning,
    get_device,
    load_shifaa_backbone,
)
from app.plots import (
    save_confusion_matrix_plot,
    save_error_visualization,
    save_training_curves,
)
from app.train import create_dataloaders, run_training_pipeline
from app.evaluate import run_evaluation_pipeline
from app.utils import ensure_dirs


def main():
    settings = get_settings()

    ensure_dirs([
        settings.output_dir,
        settings.model_dir,
        settings.plots_dir,
        settings.data_dir,
        settings.preprocessed_dir,
    ])

    print("=" * 70)
    print("INITIALISATION DU TRAIN-SERVICE")
    print("=" * 70)

    setup_mlflow(
        tracking_uri=settings.mlflow_tracking_uri,
        experiment_name=settings.mlflow_experiment_name,
    )

    print("Préparation des données...")
    prepared = prepare_data(settings)

    df_train = prepared["df_train"]
    df_val = prepared["df_val"]
    df_test = prepared["df_test"]
    label_encoder = prepared["label_encoder"]
    class_weights = prepared["class_weights"]

    print(f"Train size : {len(df_train)}")
    print(f"Val size   : {len(df_val)}")
    print(f"Test size  : {len(df_test)}")
    print(f"Classes    : {list(label_encoder.classes_)}")
    print(f"Class weights : {class_weights}")

    print("Chargement du modèle...")
    device = get_device()
    base_model = load_shifaa_backbone(num_classes=3)
    base_model = configure_model_for_finetuning(base_model)
    base_model = base_model.to(device)

    criterion, optimizer, scheduler = build_training_components(
        model=base_model,
        class_weights=class_weights,
        learning_rate=settings.learning_rate,
        scheduler_patience=settings.lr_scheduler_patience,
        scheduler_factor=settings.lr_scheduler_factor,
        min_lr=settings.min_lr,
        device=device,
    )

    print(f"Device : {device}")

    train_loader, val_loader, test_loader = create_dataloaders(
        df_train=df_train,
        df_val=df_val,
        df_test=df_test,
        batch_size=settings.batch_size,
    )

    with mlflow.start_run(run_name=settings.run_name):
        training_output = run_training_pipeline(
            model=base_model,
            train_loader=train_loader,
            val_loader=val_loader,
            criterion=criterion,
            optimizer=optimizer,
            scheduler=scheduler,
            device=device,
            settings=settings,
        )

        evaluation_output = run_evaluation_pipeline(
            model=base_model,
            test_loader=test_loader,
            df_test=df_test,
            label_encoder=label_encoder,
            device=device,
            settings=settings,
        )

        training_curves_path = save_training_curves(
            history=training_output["history"],
            settings=settings,
        )

        confusion_matrix_path = save_confusion_matrix_plot(
            confusion_matrix=evaluation_output["confusion_matrix"],
            classes=label_encoder.classes_,
            settings=settings,
        )

        error_visualization_path = save_error_visualization(
            df_test=df_test,
            y_true=evaluation_output["y_true"],
            y_pred=evaluation_output["y_pred"],
            y_pred_proba=evaluation_output["y_pred_proba"],
            errors_idx=evaluation_output["errors_idx"],
            classes=label_encoder.classes_,
            settings=settings,
        )

        log_artifact_if_exists(training_output["best_model_path"], artifact_path="checkpoints")
        log_artifact_if_exists(training_output["final_model_path"], artifact_path="checkpoints")
        log_artifact_if_exists(training_output["final_model_full_path"], artifact_path="checkpoints")
        log_artifact_if_exists(evaluation_output["results_path"], artifact_path="results")
        log_artifact_if_exists(training_curves_path, artifact_path="plots")
        log_artifact_if_exists(confusion_matrix_path, artifact_path="plots")

        if error_visualization_path is not None:
            log_artifact_if_exists(error_visualization_path, artifact_path="plots")

        log_pytorch_model(base_model)

    print("\n" + "=" * 70)
    print("FINE-TUNING TERMINÉ")
    print("=" * 70)
    print(f"Accuracy  : {evaluation_output['accuracy']*100:.2f}%")
    print(f"Precision : {evaluation_output['precision']*100:.2f}%")
    print(f"Recall    : {evaluation_output['recall']*100:.2f}%")
    print(f"F1-Score  : {evaluation_output['f1']*100:.2f}%")
    print(f"Modèle final : {training_output['final_model_full_path']}")
    print(f"Résultats    : {evaluation_output['results_path']}")
    
    


if __name__ == "__main__":
    main()