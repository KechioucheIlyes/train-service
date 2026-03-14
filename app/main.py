from app.config import get_settings
from app.data import prepare_data
from app.mlflow_utils import setup_mlflow
from app.model import (
    build_training_components,
    configure_model_for_finetuning,
    get_device,
    load_shifaa_backbone,
)
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
    print("Le scripting de base est prêt.")
    print("Étape suivante : brancher la boucle d'entraînement et l'évaluation.")


if __name__ == "__main__":
    main()