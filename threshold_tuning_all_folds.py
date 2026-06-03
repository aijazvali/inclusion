import numpy as np
import pandas as pd
from pathlib import Path
from PIL import Image

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms


# ==================================================
# 1. Configuration
# ==================================================

manifest_path = "inclusion_binary_manifest_with_splits.csv"

fold_columns = [
    "fold_test_heat_1",
    "fold_test_heat_2",
    "fold_test_heat_3",
    "fold_test_heat_4",
]

batch_size = 64

thresholds = np.arange(0.05, 0.96, 0.01)


# ==================================================
# 2. Device
# ==================================================

if torch.cuda.is_available():
    device = torch.device("cuda")
elif torch.backends.mps.is_available():
    device = torch.device("mps")
else:
    device = torch.device("cpu")

print("Using device:", device)


# ==================================================
# 3. Dataset
# ==================================================

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

        return image, label, idx


# ==================================================
# 4. Model
# ==================================================

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


# ==================================================
# 5. Metrics
# ==================================================

def compute_metrics(y_true, prob_inclusion, threshold):
    y_true = np.array(y_true)
    prob_inclusion = np.array(prob_inclusion)

    y_pred = (prob_inclusion >= threshold).astype(int)

    tp = int(((y_true == 1) & (y_pred == 1)).sum())
    tn = int(((y_true == 0) & (y_pred == 0)).sum())
    fp = int(((y_true == 0) & (y_pred == 1)).sum())
    fn = int(((y_true == 1) & (y_pred == 0)).sum())

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
        "threshold": float(threshold),
        "accuracy": accuracy,
        "balanced_accuracy": balanced_accuracy,
        "precision_inclusion": precision,
        "recall_inclusion": recall,
        "specificity_non_inclusion": specificity,
        "f1_inclusion": f1,
        "tp": tp,
        "tn": tn,
        "fp": fp,
        "fn": fn,
    }


def get_predictions(model, loader, device):
    model.eval()

    all_indices = []
    all_labels = []
    all_prob_non_inclusion = []
    all_prob_inclusion = []

    with torch.no_grad():
        for images, labels, indices in loader:
            images = images.to(device)

            outputs = model(images)
            probs = torch.softmax(outputs, dim=1)

            all_indices.extend(indices.numpy().tolist())
            all_labels.extend(labels.numpy().tolist())
            all_prob_non_inclusion.extend(probs[:, 0].cpu().numpy().tolist())
            all_prob_inclusion.extend(probs[:, 1].cpu().numpy().tolist())

    return {
        "indices": np.array(all_indices),
        "labels": np.array(all_labels),
        "prob_non_inclusion": np.array(all_prob_non_inclusion),
        "prob_inclusion": np.array(all_prob_inclusion),
    }


# ==================================================
# 6. Tune one fold
# ==================================================

def tune_one_fold(df, fold_column):
    print("\n" + "=" * 90)
    print("Threshold tuning:", fold_column)
    print("=" * 90)

    fold_number = fold_column.replace("fold_test_heat_", "")

    checkpoint_path = f"best_inclusion_cnn_fold{fold_number}.pt"

    if not Path(checkpoint_path).exists():
        raise FileNotFoundError(f"Missing checkpoint: {checkpoint_path}")

    checkpoint = torch.load(checkpoint_path, map_location=device)

    mean = checkpoint["mean"]
    std = checkpoint["std"]

    print("Loaded checkpoint:", checkpoint_path)
    print("Best epoch:", checkpoint.get("epoch", "NA"))
    print("Best val balanced accuracy:", checkpoint.get("best_val_balanced_accuracy", "NA"))
    print("Mean:", mean)
    print("Std:", std)

    val_df = df[df[fold_column] == "val"].copy().reset_index(drop=True)
    test_df = df[df[fold_column] == "test"].copy().reset_index(drop=True)

    print("\nVal samples:", len(val_df))
    print(val_df["label_name"].value_counts())

    print("\nTest samples:", len(test_df))
    print(test_df["label_name"].value_counts())

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[mean], std=[std]),
    ])

    val_dataset = InclusionDataset(val_df, transform=transform)
    test_dataset = InclusionDataset(test_df, transform=transform)

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

    model = InclusionCNN(num_classes=2).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])

    val_pred = get_predictions(model, val_loader, device)
    test_pred = get_predictions(model, test_loader, device)

    # --------------------------------------------------
    # Save raw probabilities
    # --------------------------------------------------

    val_probs_df = val_df.copy()
    val_probs_df["true_label"] = val_pred["labels"]
    val_probs_df["prob_non_inclusion"] = val_pred["prob_non_inclusion"]
    val_probs_df["prob_inclusion"] = val_pred["prob_inclusion"]

    test_probs_df = test_df.copy()
    test_probs_df["true_label"] = test_pred["labels"]
    test_probs_df["prob_non_inclusion"] = test_pred["prob_non_inclusion"]
    test_probs_df["prob_inclusion"] = test_pred["prob_inclusion"]

    val_probs_path = f"val_probabilities_fold{fold_number}.csv"
    test_probs_path = f"test_probabilities_fold{fold_number}.csv"

    val_probs_df.to_csv(val_probs_path, index=False)
    test_probs_df.to_csv(test_probs_path, index=False)

    print("\nSaved:", val_probs_path)
    print("Saved:", test_probs_path)

    # --------------------------------------------------
    # Search thresholds on validation set only
    # --------------------------------------------------

    val_rows = []

    for threshold in thresholds:
        metrics = compute_metrics(
            val_pred["labels"],
            val_pred["prob_inclusion"],
            threshold,
        )

        metrics["fold"] = fold_column
        metrics["test_heat"] = int(fold_number)
        val_rows.append(metrics)

    val_results = pd.DataFrame(val_rows)

    val_search_path = f"threshold_search_val_fold{fold_number}.csv"
    val_results.to_csv(val_search_path, index=False)

    print("Saved:", val_search_path)

    # --------------------------------------------------
    # Select thresholds using validation set
    # --------------------------------------------------

    best_bal_acc_row = val_results.sort_values(
        ["balanced_accuracy", "f1_inclusion"],
        ascending=False,
    ).iloc[0]

    best_f1_row = val_results.sort_values(
        ["f1_inclusion", "balanced_accuracy"],
        ascending=False,
    ).iloc[0]

    high_recall_candidates = val_results[val_results["recall_inclusion"] >= 0.95]

    if len(high_recall_candidates) > 0:
        high_recall_row = high_recall_candidates.sort_values(
            ["specificity_non_inclusion", "balanced_accuracy"],
            ascending=False,
        ).iloc[0]
    else:
        high_recall_row = None

    print("\nBest threshold by validation balanced accuracy:")
    print(best_bal_acc_row.to_string())

    print("\nBest threshold by validation F1:")
    print(best_f1_row.to_string())

    if high_recall_row is not None:
        print("\nBest threshold with validation recall >= 0.95:")
        print(high_recall_row.to_string())
    else:
        print("\nNo threshold achieved validation recall >= 0.95")

    # --------------------------------------------------
    # Apply selected thresholds to test set
    # --------------------------------------------------

    selected_thresholds = {
        "default_0.50": 0.50,
        "best_val_balanced_accuracy": float(best_bal_acc_row["threshold"]),
        "best_val_f1": float(best_f1_row["threshold"]),
    }

    if high_recall_row is not None:
        selected_thresholds["val_recall_at_least_0.95"] = float(high_recall_row["threshold"])

    test_rows = []

    for selection_rule, threshold in selected_thresholds.items():
        test_metrics = compute_metrics(
            test_pred["labels"],
            test_pred["prob_inclusion"],
            threshold,
        )

        test_metrics["fold"] = fold_column
        test_metrics["test_heat"] = int(fold_number)
        test_metrics["selection_rule"] = selection_rule
        test_rows.append(test_metrics)

    test_results = pd.DataFrame(test_rows)

    cols = [
        "fold",
        "test_heat",
        "selection_rule",
        "threshold",
        "accuracy",
        "balanced_accuracy",
        "precision_inclusion",
        "recall_inclusion",
        "specificity_non_inclusion",
        "f1_inclusion",
        "tp",
        "tn",
        "fp",
        "fn",
    ]

    test_results = test_results[cols]

    test_results_path = f"threshold_test_results_fold{fold_number}.csv"
    test_results.to_csv(test_results_path, index=False)

    print("\nTest results using validation-selected thresholds:")
    print(test_results.to_string(index=False))

    print("\nSaved:", test_results_path)

    selected_summary = []

    for selection_rule, threshold in selected_thresholds.items():
        val_metrics = compute_metrics(
            val_pred["labels"],
            val_pred["prob_inclusion"],
            threshold,
        )

        test_metrics = compute_metrics(
            test_pred["labels"],
            test_pred["prob_inclusion"],
            threshold,
        )

        row = {
            "fold": fold_column,
            "test_heat": int(fold_number),
            "selection_rule": selection_rule,
            "threshold": threshold,

            "val_accuracy": val_metrics["accuracy"],
            "val_balanced_accuracy": val_metrics["balanced_accuracy"],
            "val_precision_inclusion": val_metrics["precision_inclusion"],
            "val_recall_inclusion": val_metrics["recall_inclusion"],
            "val_specificity_non_inclusion": val_metrics["specificity_non_inclusion"],
            "val_f1_inclusion": val_metrics["f1_inclusion"],
            "val_tp": val_metrics["tp"],
            "val_tn": val_metrics["tn"],
            "val_fp": val_metrics["fp"],
            "val_fn": val_metrics["fn"],

            "test_accuracy": test_metrics["accuracy"],
            "test_balanced_accuracy": test_metrics["balanced_accuracy"],
            "test_precision_inclusion": test_metrics["precision_inclusion"],
            "test_recall_inclusion": test_metrics["recall_inclusion"],
            "test_specificity_non_inclusion": test_metrics["specificity_non_inclusion"],
            "test_f1_inclusion": test_metrics["f1_inclusion"],
            "test_tp": test_metrics["tp"],
            "test_tn": test_metrics["tn"],
            "test_fp": test_metrics["fp"],
            "test_fn": test_metrics["fn"],
        }

        selected_summary.append(row)

    return selected_summary


# ==================================================
# 7. Main
# ==================================================

df = pd.read_csv(manifest_path)
df.columns = df.columns.astype(str).str.strip()

required_columns = [
    "sample_id",
    "heat",
    "part",
    "image_path",
    "label",
    "label_name",
] + fold_columns

missing_columns = [c for c in required_columns if c not in df.columns]

if missing_columns:
    raise ValueError(f"Missing columns in manifest: {missing_columns}")

print("\nChecking image paths...")

for p in df["image_path"].head(5):
    if not Path(p).exists():
        raise FileNotFoundError(f"Image path not found: {p}")

print("Image path check passed.")

all_selected_rows = []

for fold_column in fold_columns:
    rows = tune_one_fold(df, fold_column)
    all_selected_rows.extend(rows)

summary = pd.DataFrame(all_selected_rows)

summary_path = "threshold_tuning_all_folds_summary.csv"
summary.to_csv(summary_path, index=False)

print("\n" + "=" * 90)
print("THRESHOLD TUNING ALL FOLDS COMPLETE")
print("=" * 90)

print("\nFull summary:")
print(summary.to_string(index=False))

print("\nAverage test metrics by selection rule:")

avg_cols = [
    "test_accuracy",
    "test_balanced_accuracy",
    "test_precision_inclusion",
    "test_recall_inclusion",
    "test_specificity_non_inclusion",
    "test_f1_inclusion",
    "test_fp",
    "test_fn",
]

avg_summary = summary.groupby("selection_rule")[avg_cols].mean().reset_index()
print(avg_summary.to_string(index=False))

avg_summary.to_csv("threshold_tuning_average_by_rule.csv", index=False)

print("\nSaved:", summary_path)
print("Saved: threshold_tuning_average_by_rule.csv")