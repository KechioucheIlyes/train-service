from copy import deepcopy
import os

import torch
from torch.utils.data import DataLoader

from app.dataset import CovidTorchDataset
from app.mlflow_utils import log_epoch_metrics, log_training_params
from app.utils import ensure_dir


def create_dataloaders(df_train, df_val, df_test, batch_size: int):
    train_dataset = CovidTorchDataset(df_train)
    val_dataset = CovidTorchDataset(df_val)
    test_dataset = CovidTorchDataset(df_test)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader, test_loader


def run_training_pipeline(
    model,
    train_loader,
    val_loader,
    criterion,
    optimizer,
    scheduler,
    device,
    settings,
):
    ensure_dir(settings.model_dir)

    best_val_loss = float("inf")
    best_val_acc = 0.0
    epochs_without_improvement = 0
    best_model_weights = None

    best_model_path = os.path.join(settings.model_dir, "best_model.pth")
    final_model_path = os.path.join(settings.model_dir, "final_model.pth")
    final_model_full_path = os.path.join(settings.model_dir, "final_model_full.pth")

    history = {
        "train_loss": [],
        "train_acc": [],
        "val_loss": [],
        "val_acc": [],
    }

    log_training_params(
        model_type="Shifaa_ResNet50_PyTorch",
        num_classes=3,
        epochs=settings.epochs,
        batch_size=settings.batch_size,
        learning_rate=settings.learning_rate,
        dataset_name="S7",
        img_size=settings.img_size,
    )

    print("\n" + "=" * 70)
    print("DEBUT DE L'ENTRAINEMENT")
    print("=" * 70)

    for epoch in range(settings.epochs):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0

        for images, labels in train_loader:
            images = images.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * images.size(0)
            _, preds = torch.max(outputs, 1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

        train_loss = running_loss / total
        train_acc = correct / total

        model.eval()
        running_val_loss = 0.0
        correct_val = 0
        total_val = 0

        with torch.no_grad():
            for images, labels in val_loader:
                images = images.to(device)
                labels = labels.to(device)

                outputs = model(images)
                loss = criterion(outputs, labels)

                running_val_loss += loss.item() * images.size(0)
                _, preds = torch.max(outputs, 1)
                correct_val += (preds == labels).sum().item()
                total_val += labels.size(0)

        val_loss = running_val_loss / total_val
        val_acc = correct_val / total_val

        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)

        log_epoch_metrics(
            epoch=epoch,
            train_loss=train_loss,
            train_acc=train_acc,
            val_loss=val_loss,
            val_acc=val_acc,
        )

        print(
            f"Epoch [{epoch+1}/{settings.epochs}] | "
            f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f} | "
            f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f}"
        )

        scheduler.step(val_loss)

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_model_weights = deepcopy(model.state_dict())
            torch.save(model.state_dict(), best_model_path)
            print(f"Meilleur modèle sauvegardé : {best_model_path}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1

        if epochs_without_improvement >= settings.early_stopping_patience:
            print(f"\n Early stopping déclenché après {epoch+1} epochs")
            break

    if best_model_weights is not None:
        model.load_state_dict(best_model_weights)

    torch.save(model.state_dict(), final_model_path)
    torch.save(model, final_model_full_path)

    

    return {
        "history": history,
        "best_model_path": best_model_path,
        "final_model_path": final_model_path,
        "final_model_full_path": final_model_full_path,
        "best_val_acc": best_val_acc,
        "best_val_loss": best_val_loss,
    }