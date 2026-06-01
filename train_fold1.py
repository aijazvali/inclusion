import random
import numpy as np
import pandas as pd
from pathlib import Path
from PIL import Image

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms


# --------------------------------------------------
# 1. Configuration
# --------------------------------------------------

manifest_path = "inclusion_binary_manifest_with_splits.csv"
fold_column = "fold_test_heat_1"

batch_size = 64
num_epochs = 15
learning_rate = 1e-3
weight_decay = 1e-4
seed = 42

best_model_path = "best_inclusion_cnn_fold1.pt"
log_path = "training_log_fold1.csv"


# --------------------------------------------------
# 2. Reproducibility
# --------------------------------------------------

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

set_seed(seed)


# --------------------------------------------------
# 3. Device
# --------------------------------------------------

if torch.cuda.is_available():
    device = torch.device("cuda")
elif torch.backends.mps.is_available():
    device = torch.device("mps")
else:
    device = torch.device("cpu")

print("Using device:", device)


# --------------------------------------------------
# 4. Dataset
# --------------------------------------------------

class InclusionDataset(Dataset):
    def __init__(self, dataframe, transform=None):
        self.df = dataframe.reset_index(drop=True)
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]

        image = Image.open(row["image_path"]).convert("L")
        label = int(row["label"])

        if self.transform is not None:
            image = self.transform(image)

        label = torch.tensor(label, dtype=torch.long)

        return image, label


# --------------------------------------------------
# 5. Model
# --------------------------------------------------

class InclusionCNN(nn.Module):
    def __init__(self, num_classes=2):
        super().__init__()

        self.features = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),

            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),

            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),

            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),

            nn.Conv2d(256, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((1, 1)),
        )

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(0.3),
            nn.Linear(512, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(128, num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x


# --------------------------------------------------
# 6. Metrics
# --------------------------------------------------

def compute_metrics_from_counts(tp, tn, fp, fn):
    total = tp + tn + fp + fn

    accuracy = (tp + tn) / total if total > 0 else 0.0

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0

    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0

    f1 = (
        2 * precision * recall / (precision + recall)
        if (precision + recall) > 0
        else 0.0
    )

    balanced_accuracy = (recall + specificity) / 2

    return {
        "accuracy": accuracy,
        "precision_inclusion": precision,
        "recall_inclusion": recall,
        "specificity_non_inclusion": specificity,
        "f1_inclusion": f1,
        "balanced_accuracy": balanced_accuracy,
        "tp": tp,
        "tn": tn,
        "fp": fp,
        "fn": fn,
    }


def evaluate(model, loader, criterion, device):
    model.eval()

    total_loss = 0.0
    total_samples = 0

    tp = tn = fp = fn = 0

    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)
            loss = criterion(outputs, labels)

            preds = outputs.argmax(dim=1)

            total_loss += loss.item() * labels.size(0)
            total_samples += labels.size(0)

            tp += ((preds == 1) & (labels == 1)).sum().item()
            tn += ((preds == 0) & (labels == 0)).sum().item()
            fp += ((preds == 1) & (labels == 0)).sum().item()
            fn += ((preds == 0) & (labels == 1)).sum().item()

    avg_loss = total_loss / total_samples
    metrics = compute_metrics_from_counts(tp, tn, fp, fn)
    metrics["loss"] = avg_loss

    return metrics


# --------------------------------------------------
# 7. Load data
# --------------------------------------------------

df = pd.read_csv(manifest_path)

train_df = df[df[fold_column] == "train"].copy()
val_df = df[df[fold_column] == "val"].copy()
test_df = df[df[fold_column] == "test"].copy()

print("Using fold:", fold_column)
print("Train samples:", len(train_df))
print("Val samples:", len(val_df))
print("Test samples:", len(test_df))

print("\nTrain class balance:")
print(train_df["label_name"].value_counts())

print("\nVal class balance:")
print(val_df["label_name"].value_counts())

print("\nTest class balance:")
print(test_df["label_name"].value_counts())


# --------------------------------------------------
# 8. Training-set normalization
# --------------------------------------------------

basic_transform = transforms.Compose([
    transforms.ToTensor(),
])

basic_train_dataset = InclusionDataset(train_df, transform=basic_transform)

basic_train_loader = DataLoader(
    basic_train_dataset,
    batch_size=batch_size,
    shuffle=False,
    num_workers=0,
)


def compute_mean_std(loader):
    total_sum = 0.0
    total_squared_sum = 0.0
    total_pixels = 0

    for images, _ in loader:
        total_sum += images.sum().item()
        total_squared_sum += (images ** 2).sum().item()
        total_pixels += images.numel()

    mean = total_sum / total_pixels
    variance = (total_squared_sum / total_pixels) - (mean ** 2)
    std = variance ** 0.5

    return mean, std


mean, std = compute_mean_std(basic_train_loader)

print("\nTraining mean:", mean)
print("Training std:", std)


# --------------------------------------------------
# 9. Transforms
# --------------------------------------------------

train_transform = transforms.Compose([
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomVerticalFlip(p=0.5),
    transforms.ToTensor(),
    transforms.Normalize(mean=[mean], std=[std]),
])

eval_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[mean], std=[std]),
])

train_dataset = InclusionDataset(train_df, transform=train_transform)
val_dataset = InclusionDataset(val_df, transform=eval_transform)
test_dataset = InclusionDataset(test_df, transform=eval_transform)

train_loader = DataLoader(
    train_dataset,
    batch_size=batch_size,
    shuffle=True,
    num_workers=0,
)

val_loader = DataLoader(
    val_dataset,
    batch_size=batch_size,
    shuffle=False,
    num_workers=0,
)

test_loader = DataLoader(
    test_dataset,
    batch_size=batch_size,
    shuffle=False,
    num_workers=0,
)


# --------------------------------------------------
# 10. Loss, model, optimizer
# --------------------------------------------------

model = InclusionCNN(num_classes=2).to(device)

class_counts = train_df["label"].value_counts().sort_index()
num_non_inclusion = class_counts[0]
num_inclusion = class_counts[1]

total_train = num_non_inclusion + num_inclusion

weight_non_inclusion = total_train / (2 * num_non_inclusion)
weight_inclusion = total_train / (2 * num_inclusion)

class_weights = torch.tensor(
    [weight_non_inclusion, weight_inclusion],
    dtype=torch.float32,
).to(device)

print("\nClass weights:")
print("non-inclusion:", weight_non_inclusion)
print("inclusion:", weight_inclusion)

criterion = nn.CrossEntropyLoss(weight=class_weights)

optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=learning_rate,
    weight_decay=weight_decay,
)


# --------------------------------------------------
# 11. Training loop
# --------------------------------------------------

best_val_balanced_accuracy = 0.0
history = []

for epoch in range(1, num_epochs + 1):
    model.train()

    running_loss = 0.0
    total_samples = 0

    tp = tn = fp = fn = 0

    for images, labels in train_loader:
        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()

        outputs = model(images)
        loss = criterion(outputs, labels)

        loss.backward()
        optimizer.step()

        preds = outputs.argmax(dim=1)

        running_loss += loss.item() * labels.size(0)
        total_samples += labels.size(0)

        tp += ((preds == 1) & (labels == 1)).sum().item()
        tn += ((preds == 0) & (labels == 0)).sum().item()
        fp += ((preds == 1) & (labels == 0)).sum().item()
        fn += ((preds == 0) & (labels == 1)).sum().item()

    train_metrics = compute_metrics_from_counts(tp, tn, fp, fn)
    train_metrics["loss"] = running_loss / total_samples

    val_metrics = evaluate(model, val_loader, criterion, device)

    row = {
        "epoch": epoch,

        "train_loss": train_metrics["loss"],
        "train_accuracy": train_metrics["accuracy"],
        "train_balanced_accuracy": train_metrics["balanced_accuracy"],
        "train_f1_inclusion": train_metrics["f1_inclusion"],
        "train_recall_inclusion": train_metrics["recall_inclusion"],
        "train_specificity_non_inclusion": train_metrics["specificity_non_inclusion"],

        "val_loss": val_metrics["loss"],
        "val_accuracy": val_metrics["accuracy"],
        "val_balanced_accuracy": val_metrics["balanced_accuracy"],
        "val_f1_inclusion": val_metrics["f1_inclusion"],
        "val_recall_inclusion": val_metrics["recall_inclusion"],
        "val_specificity_non_inclusion": val_metrics["specificity_non_inclusion"],
        "val_tp": val_metrics["tp"],
        "val_tn": val_metrics["tn"],
        "val_fp": val_metrics["fp"],
        "val_fn": val_metrics["fn"],
    }

    history.append(row)

    print(
        f"Epoch [{epoch:02d}/{num_epochs}] "
        f"train_loss={row['train_loss']:.4f} "
        f"train_bal_acc={row['train_balanced_accuracy']:.4f} "
        f"val_loss={row['val_loss']:.4f} "
        f"val_acc={row['val_accuracy']:.4f} "
        f"val_bal_acc={row['val_balanced_accuracy']:.4f} "
        f"val_f1={row['val_f1_inclusion']:.4f}"
    )

    if val_metrics["balanced_accuracy"] > best_val_balanced_accuracy:
        best_val_balanced_accuracy = val_metrics["balanced_accuracy"]

        checkpoint = {
            "model_state_dict": model.state_dict(),
            "mean": mean,
            "std": std,
            "fold_column": fold_column,
            "epoch": epoch,
            "best_val_balanced_accuracy": best_val_balanced_accuracy,
        }

        torch.save(checkpoint, best_model_path)

        print(f"Saved new best model: {best_model_path}")


# --------------------------------------------------
# 12. Save training log
# --------------------------------------------------

history_df = pd.DataFrame(history)
history_df.to_csv(log_path, index=False)

print("\nSaved training log:", log_path)
print("Best validation balanced accuracy:", best_val_balanced_accuracy)


# --------------------------------------------------
# 13. Evaluate best model on test set once
# --------------------------------------------------

checkpoint = torch.load(best_model_path, map_location=device)

model.load_state_dict(checkpoint["model_state_dict"])

test_metrics = evaluate(model, test_loader, criterion, device)

print("\nFinal test metrics using best validation model:")
for key, value in test_metrics.items():
    print(f"{key}: {value}")

pd.DataFrame([test_metrics]).to_csv("test_metrics_fold1.csv", index=False)
print("\nSaved test metrics: test_metrics_fold1.csv")