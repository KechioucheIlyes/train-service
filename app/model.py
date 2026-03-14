from copy import deepcopy

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau


def load_shifaa_backbone(num_classes: int = 3):
    from shifaa.vision import VisionModelFactory

    shifaa_model = VisionModelFactory.create_model(
        model_type="classification",
        model_name="Chest_COVID",
    )

    base_model = shifaa_model.model.model
    in_features = base_model.fc.in_features
    base_model.fc = nn.Linear(in_features, num_classes)

    return base_model


def configure_model_for_finetuning(base_model: torch.nn.Module) -> torch.nn.Module:
    for param in base_model.parameters():
        param.requires_grad = False

    for param in base_model.fc.parameters():
        param.requires_grad = True

    return base_model


def build_training_components(
    model: torch.nn.Module,
    class_weights: dict[int, float],
    learning_rate: float,
    scheduler_patience: int,
    scheduler_factor: float,
    min_lr: float,
    device: torch.device,
):
    class_weights_tensor = torch.tensor(
        [class_weights[i] for i in range(len(class_weights))],
        dtype=torch.float32,
    ).to(device)

    criterion = nn.CrossEntropyLoss(weight=class_weights_tensor)
    optimizer = optim.Adam(model.fc.parameters(), lr=learning_rate)

    scheduler = ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=scheduler_factor,
        patience=scheduler_patience,
        min_lr=min_lr,
    )

    return criterion, optimizer, scheduler


def get_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")