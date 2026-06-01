import numpy as np
import pandas as pd
from PIL import Image

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms


# --------------------------------------------------
# 1. Configuration
# --------------------------------------------------

manifest_path = "inclusion_binary_manifest_with_splits.csv"
checkpoint_path = "best_inclusion_cnn_fold1.pt"
fold_column = "fold_test_heat_1"

batch_size = 64


# --------------------------------------------------
# 2. Device
# --------------------------------------------------

if torch.cuda.is_available():
    device = torch.device("cuda")
elif torch.backends.mps.is_available():
    device = torch.device("mps")
else:
    device = torch.device("cpu")

print("Using device:", device)


# --------------------------------------------------
# 3. Dataset
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

        return image, label


# --------------------------------------------------
# 4. Model
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
# 5. Metric function
# --------------------------------------------------

def compute_metrics(y_true, prob_inclusion, threshold):
    y_true = np.array(y_true)
    prob_inclusion = np.array(prob_inclusion)

    y_pred = (prob_inclusion >= threshold).astype(int)

    tp = ((y_true == 1) & (y_pred == 1)).sum()
    tn = ((y_true == 0) & (y_pred == 0)).sum()
    fp = ((y_true == 0) & (y_pred == 1)).sum()
    fn = ((y_true == 1) & (y_pred == 0)).sum()

    total = tp + tn + fp + fn

    accuracy = (tp + tn) / total if total > 0 else 0

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0

    f1 = (
        2 * precision * recall / (precision + recall)
        if (precision + recall) > 0
        else 0
    )

    balanced_accuracy = (recall + specificity) / 2

    return {
        "threshold": threshold,
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


# --------------------------------------------------
# 6. Prediction function
# --------------------------------------------------

def get_predictions(model, loader, device):
    model.eval()

    all_labels = []
    all_prob_inclusion = []

    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device)

            outputs = model(images)
            probs = torch.softmax(outputs, dim=1)

            prob_inclusion = probs[:, 1].cpu().numpy()

            all_labels.extend(labels.numpy().tolist())
            all_prob_inclusion.extend(prob_inclusion.tolist())

    return np.array(all_labels), np.array(all_prob_inclusion)


# --------------------------------------------------
# 7. Load checkpoint and data
# --------------------------------------------------

checkpoint = torch.load(checkpoint_path, map_location=device)

mean = checkpoint["mean"]
std = checkpoint["std"]

print("Loaded checkpoint:", checkpoint_path)
print("Best epoch:", checkpoint["epoch"])
print("Best val balanced accuracy:", checkpoint["best_val_balanced_accuracy"])
print("Mean:", mean)
print("Std:", std)

df = pd.read_csv(manifest_path)

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


# --------------------------------------------------
# 8. Load model
# --------------------------------------------------

model = InclusionCNN(num_classes=2).to(device)
model.load_state_dict(checkpoint["model_state_dict"])


# --------------------------------------------------
# 9. Predict validation and test sets
# --------------------------------------------------

val_true, val_prob = get_predictions(model, val_loader, device)
test_true, test_prob = get_predictions(model, test_loader, device)


# --------------------------------------------------
# 10. Try many thresholds on validation set
# --------------------------------------------------

thresholds = np.arange(0.05, 0.96, 0.01)

val_rows = []

for threshold in thresholds:
    metrics = compute_metrics(val_true, val_prob, threshold)
    val_rows.append(metrics)

val_results = pd.DataFrame(val_rows)

val_results.to_csv("threshold_search_val_fold1.csv", index=False)

print("\nSaved: threshold_search_val_fold1.csv")


# --------------------------------------------------
# 11. Select thresholds using validation set only
# --------------------------------------------------

best_bal_acc_row = val_results.sort_values(
    "balanced_accuracy",
    ascending=False
).iloc[0]

best_f1_row = val_results.sort_values(
    "f1_inclusion",
    ascending=False
).iloc[0]

# Safety-oriented option:
# among thresholds with recall >= 0.95, choose best specificity
high_recall_candidates = val_results[val_results["recall_inclusion"] >= 0.95]

if len(high_recall_candidates) > 0:
    high_recall_row = high_recall_candidates.sort_values(
        "specificity_non_inclusion",
        ascending=False
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
# 12. Apply selected thresholds to test set
# --------------------------------------------------

selected = {
    "default_0.50": 0.50,
    "best_val_balanced_accuracy": float(best_bal_acc_row["threshold"]),
    "best_val_f1": float(best_f1_row["threshold"]),
}

if high_recall_row is not None:
    selected["val_recall_at_least_0.95"] = float(high_recall_row["threshold"])

test_rows = []

for name, threshold in selected.items():
    metrics = compute_metrics(test_true, test_prob, threshold)
    metrics["selection_rule"] = name
    test_rows.append(metrics)

test_results = pd.DataFrame(test_rows)

# Move selection_rule to first column
cols = ["selection_rule"] + [c for c in test_results.columns if c != "selection_rule"]
test_results = test_results[cols]

test_results.to_csv("threshold_test_results_fold1.csv", index=False)

print("\nTest results using validation-selected thresholds:")
print(test_results.to_string(index=False))

print("\nSaved: threshold_test_results_fold1.csv")

print("\nStep 9 threshold tuning complete.")