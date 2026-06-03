import os
import copy
import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    f1_score,
    classification_report,
    confusion_matrix,
)

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader


# ----------------------------
# CONFIG
# ----------------------------

INPUT_CSV = "phase_features_labeled.csv"
OUTPUT_DIR = "outputs_image_plus_phase_features_type_classifier_3class"

TARGET_COL = "type"
HEAT_COL = "heat"
INCLUSION_COL = "inclusion"

DROP_CLASSES = ["other", "nitride", "non-inclusion", "non-inc", "non inclusion"]

IMG_SIZE = 128
BATCH_SIZE = 128
EPOCHS = 15
LR = 1e-3
VAL_SIZE = 0.15
RANDOM_STATE = 42

os.makedirs(OUTPUT_DIR, exist_ok=True)

try:
    cv2.setLogLevel(0)
except Exception:
    pass


# ----------------------------
# DEVICE
# ----------------------------

if torch.backends.mps.is_available():
    DEVICE = torch.device("mps")
elif torch.cuda.is_available():
    DEVICE = torch.device("cuda")
else:
    DEVICE = torch.device("cpu")

print("Using device:", DEVICE)


# ----------------------------
# HELPERS
# ----------------------------

def clean_label_series(s):
    return (
        s.astype(str)
        .str.strip()
        .str.replace("_", "-", regex=False)
        .str.lower()
    )


def normalize_image(img):
    img = img.astype(np.float32)

    low, high = np.percentile(img, [1, 99])

    if high <= low:
        img = img / 255.0
        return np.clip(img, 0, 1)

    img = (img - low) / (high - low)
    img = np.clip(img, 0, 1)

    return img


def read_image(path):
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)

    if img is None:
        raise ValueError(f"Could not read image: {path}")

    if img.shape[0] != IMG_SIZE or img.shape[1] != IMG_SIZE:
        img = cv2.resize(img, (IMG_SIZE, IMG_SIZE), interpolation=cv2.INTER_AREA)

    img = normalize_image(img)

    return img


def get_feature_columns(df):
    allowed_exact = [
        "image_h",
        "image_w",
        "image_area",
        "particle_area",
        "particle_area_fraction",
        "bbox_x",
        "bbox_y",
        "bbox_w",
        "bbox_h",
        "bbox_aspect",
        "bbox_extent",
        "particle_mean_intensity",
        "particle_std_intensity",
        "particle_min_intensity",
        "particle_max_intensity",
        "total_phase_regions",
    ]

    allowed_prefixes = [
        "phase_1_",
        "phase_2_",
        "phase_3_",
    ]

    feature_cols = []

    for col in df.columns:
        if col in allowed_exact:
            feature_cols.append(col)
        elif any(col.startswith(prefix) for prefix in allowed_prefixes):
            feature_cols.append(col)

    numeric_cols = []

    for col in feature_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")
        if pd.api.types.is_numeric_dtype(df[col]):
            numeric_cols.append(col)

    return numeric_cols


def plot_confusion_matrix(cm, labels, save_path, title):
    fig, ax = plt.subplots(figsize=(7, 6))

    im = ax.imshow(cm)

    ax.set_title(title)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")

    ax.set_xticks(np.arange(len(labels)))
    ax.set_yticks(np.arange(len(labels)))

    ax.set_xticklabels(labels, rotation=45, ha="right")
    ax.set_yticklabels(labels)

    for i in range(len(labels)):
        for j in range(len(labels)):
            ax.text(j, i, str(cm[i, j]), ha="center", va="center")

    fig.colorbar(im, ax=ax)
    plt.tight_layout()
    plt.savefig(save_path, dpi=200)
    plt.close()


# ----------------------------
# DATASET
# ----------------------------

class InclusionCombinedDataset(Dataset):
    def __init__(self, image_paths, tab_features, labels, train=False):
        self.image_paths = image_paths
        self.tab_features = tab_features
        self.labels = labels
        self.train = train

    def __len__(self):
        return len(self.image_paths)

    def augment(self, img):
        if np.random.rand() < 0.5:
            img = np.fliplr(img)

        if np.random.rand() < 0.5:
            img = np.flipud(img)

        k = np.random.randint(0, 4)
        img = np.rot90(img, k)

        return img.copy()

    def __getitem__(self, idx):
        img = read_image(self.image_paths[idx])

        if self.train:
            img = self.augment(img)

        img = torch.tensor(img, dtype=torch.float32).unsqueeze(0)

        tab = torch.tensor(self.tab_features[idx], dtype=torch.float32)

        label = torch.tensor(self.labels[idx], dtype=torch.long)

        return img, tab, label


# ----------------------------
# MODEL
# ----------------------------

class CombinedCNNTabular(nn.Module):
    def __init__(self, num_tab_features, num_classes):
        super().__init__()

        self.image_encoder = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),

            nn.Conv2d(16, 32, kernel_size=3, padding=1),
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

            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
        )

        self.tabular_encoder = nn.Sequential(
            nn.Linear(num_tab_features, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.20),

            nn.Linear(128, 64),
            nn.ReLU(inplace=True),
        )

        self.classifier = nn.Sequential(
            nn.Linear(256 + 64, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.35),
            nn.Linear(128, num_classes),
        )

    def forward(self, img, tab):
        img_feat = self.image_encoder(img)
        tab_feat = self.tabular_encoder(tab)

        combined = torch.cat([img_feat, tab_feat], dim=1)

        logits = self.classifier(combined)

        return logits


# ----------------------------
# TRAIN / EVAL
# ----------------------------

def train_one_epoch(model, loader, criterion, optimizer):
    model.train()

    total_loss = 0.0
    all_preds = []
    all_true = []

    for imgs, tabs, labels in loader:
        imgs = imgs.to(DEVICE)
        tabs = tabs.to(DEVICE)
        labels = labels.to(DEVICE)

        optimizer.zero_grad()

        logits = model(imgs, tabs)
        loss = criterion(logits, labels)

        loss.backward()
        optimizer.step()

        total_loss += loss.item() * imgs.size(0)

        preds = torch.argmax(logits, dim=1)

        all_preds.extend(preds.detach().cpu().tolist())
        all_true.extend(labels.detach().cpu().tolist())

    avg_loss = total_loss / len(loader.dataset)
    acc = accuracy_score(all_true, all_preds)

    return avg_loss, acc


@torch.no_grad()
def evaluate(model, loader, criterion):
    model.eval()

    total_loss = 0.0
    all_preds = []
    all_true = []

    for imgs, tabs, labels in loader:
        imgs = imgs.to(DEVICE)
        tabs = tabs.to(DEVICE)
        labels = labels.to(DEVICE)

        logits = model(imgs, tabs)
        loss = criterion(logits, labels)

        total_loss += loss.item() * imgs.size(0)

        preds = torch.argmax(logits, dim=1)

        all_preds.extend(preds.detach().cpu().tolist())
        all_true.extend(labels.detach().cpu().tolist())

    avg_loss = total_loss / len(loader.dataset)

    return avg_loss, np.array(all_true), np.array(all_preds)


# ----------------------------
# LOAD DATA
# ----------------------------

df = pd.read_csv(INPUT_CSV)

print("Loaded:", INPUT_CSV)
print("Shape:", df.shape)

df[TARGET_COL] = clean_label_series(df[TARGET_COL])
df = df[~df[TARGET_COL].isin(["nan", "", "none"])].copy()

if INCLUSION_COL in df.columns:
    df[INCLUSION_COL] = pd.to_numeric(df[INCLUSION_COL], errors="coerce")
    df = df[df[INCLUSION_COL] == 1].copy()

df = df[~df[TARGET_COL].isin(DROP_CLASSES)].copy()

df[HEAT_COL] = pd.to_numeric(df[HEAT_COL], errors="coerce")
df = df.dropna(subset=[HEAT_COL])
df[HEAT_COL] = df[HEAT_COL].astype(int)

df["image_exists"] = df["image_path"].apply(os.path.exists)
missing = (~df["image_exists"]).sum()

if missing > 0:
    print("Missing image files:", missing)

df = df[df["image_exists"]].copy()

feature_cols = get_feature_columns(df)

print("\nNumber of phase features:", len(feature_cols))
for col in feature_cols:
    print(" -", col)

classes = sorted(df[TARGET_COL].unique())
label_to_idx = {c: i for i, c in enumerate(classes)}
idx_to_label = {i: c for c, i in label_to_idx.items()}

print("\nClasses:", classes)
print("\nClass counts:")
print(df[TARGET_COL].value_counts())

heats = sorted(df[HEAT_COL].unique())
print("\nHeat-wise folds:", heats)


# ----------------------------
# HEAT-WISE TRAINING
# ----------------------------

all_metrics = []
all_predictions = []

for test_heat in heats:
    print("\n" + "=" * 60)
    print(f"Fold: Test Heat {test_heat}")
    print("=" * 60)

    trainval_df = df[df[HEAT_COL] != test_heat].copy()
    test_df = df[df[HEAT_COL] == test_heat].copy()

    train_df, val_df = train_test_split(
        trainval_df,
        test_size=VAL_SIZE,
        random_state=RANDOM_STATE,
        stratify=trainval_df[TARGET_COL],
    )

    print("Train size:", len(train_df))
    print("Val size:", len(val_df))
    print("Test size:", len(test_df))

    print("\nTrain class counts:")
    print(train_df[TARGET_COL].value_counts())

    print("\nTest class counts:")
    print(test_df[TARGET_COL].value_counts())

    # tabular preprocessing: fit only on train
    imputer = SimpleImputer(strategy="median")
    scaler = StandardScaler()

    X_train_tab = imputer.fit_transform(train_df[feature_cols])
    X_val_tab = imputer.transform(val_df[feature_cols])
    X_test_tab = imputer.transform(test_df[feature_cols])

    X_train_tab = scaler.fit_transform(X_train_tab)
    X_val_tab = scaler.transform(X_val_tab)
    X_test_tab = scaler.transform(X_test_tab)

    y_train = train_df[TARGET_COL].map(label_to_idx).astype(int).tolist()
    y_val = val_df[TARGET_COL].map(label_to_idx).astype(int).tolist()
    y_test = test_df[TARGET_COL].map(label_to_idx).astype(int).tolist()

    train_dataset = InclusionCombinedDataset(
        train_df["image_path"].tolist(),
        X_train_tab.tolist(),
        y_train,
        train=True,
    )

    val_dataset = InclusionCombinedDataset(
        val_df["image_path"].tolist(),
        X_val_tab.tolist(),
        y_val,
        train=False,
    )

    test_dataset = InclusionCombinedDataset(
        test_df["image_path"].tolist(),
        X_test_tab.tolist(),
        y_test,
        train=False,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=0,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0,
    )

    model = CombinedCNNTabular(
        num_tab_features=len(feature_cols),
        num_classes=len(classes),
    ).to(DEVICE)

    train_counts = train_df[TARGET_COL].value_counts()
    weights = []

    for c in classes:
        weights.append(len(train_df) / (len(classes) * train_counts[c]))

    weights = torch.tensor(weights, dtype=torch.float32).to(DEVICE)

    criterion = nn.CrossEntropyLoss(weight=weights)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)

    best_val_loss = float("inf")
    best_state = None
    history_rows = []

    for epoch in range(1, EPOCHS + 1):
        train_loss, train_acc = train_one_epoch(
            model,
            train_loader,
            criterion,
            optimizer,
        )

        val_loss, val_true_idx, val_pred_idx = evaluate(
            model,
            val_loader,
            criterion,
        )

        val_acc = accuracy_score(val_true_idx, val_pred_idx)
        val_macro_f1 = f1_score(val_true_idx, val_pred_idx, average="macro")

        history_rows.append(
            {
                "test_heat": test_heat,
                "epoch": epoch,
                "train_loss": train_loss,
                "train_acc": train_acc,
                "val_loss": val_loss,
                "val_acc": val_acc,
                "val_macro_f1": val_macro_f1,
            }
        )

        print(
            f"Epoch {epoch:02d}/{EPOCHS} | "
            f"train_loss={train_loss:.4f} train_acc={train_acc:.4f} | "
            f"val_loss={val_loss:.4f} val_acc={val_acc:.4f} val_macro_f1={val_macro_f1:.4f}"
        )

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = copy.deepcopy(model.state_dict())

    if best_state is not None:
        model.load_state_dict(best_state)

    history_df = pd.DataFrame(history_rows)
    history_path = os.path.join(
        OUTPUT_DIR,
        f"training_history_combined_test_heat_{test_heat}.csv"
    )
    history_df.to_csv(history_path, index=False)

    test_loss, y_true_idx, y_pred_idx = evaluate(
        model,
        test_loader,
        criterion,
    )

    y_true = [idx_to_label[i] for i in y_true_idx]
    y_pred = [idx_to_label[i] for i in y_pred_idx]

    acc = accuracy_score(y_true, y_pred)
    bal_acc = balanced_accuracy_score(y_true, y_pred)
    macro_f1 = f1_score(y_true, y_pred, average="macro")
    weighted_f1 = f1_score(y_true, y_pred, average="weighted")

    metrics_row = {
        "test_heat": test_heat,
        "train_size": len(train_df),
        "val_size": len(val_df),
        "test_size": len(test_df),
        "test_loss": test_loss,
        "accuracy": acc,
        "balanced_accuracy": bal_acc,
        "macro_f1": macro_f1,
        "weighted_f1": weighted_f1,
    }

    all_metrics.append(metrics_row)

    print("\nTest metrics:")
    for k, v in metrics_row.items():
        print(f"{k}: {v}")

    report = classification_report(
        y_true,
        y_pred,
        labels=classes,
        zero_division=0,
    )

    report_path = os.path.join(
        OUTPUT_DIR,
        f"classification_report_combined_test_heat_{test_heat}.txt"
    )

    with open(report_path, "w") as f:
        f.write(report)

    print("\nClassification report:")
    print(report)

    cm = confusion_matrix(y_true, y_pred, labels=classes)

    cm_path = os.path.join(
        OUTPUT_DIR,
        f"confusion_matrix_combined_test_heat_{test_heat}.png"
    )

    plot_confusion_matrix(
        cm,
        classes,
        cm_path,
        title=f"Image + Phase Features - Test Heat {test_heat}",
    )

    pred_df = test_df[["image_path", "mask_path", HEAT_COL, "particle_id", TARGET_COL]].copy()
    pred_df = pred_df.rename(columns={TARGET_COL: "true_type"})
    pred_df["pred_type"] = y_pred
    pred_df["test_heat"] = test_heat

    all_predictions.append(pred_df)

    model_path = os.path.join(
        OUTPUT_DIR,
        f"combined_model_test_heat_{test_heat}.pt"
    )
    torch.save(model.state_dict(), model_path)


# ----------------------------
# SAVE RESULTS
# ----------------------------

metrics_df = pd.DataFrame(all_metrics)
metrics_path = os.path.join(OUTPUT_DIR, "combined_type_fold_metrics.csv")
metrics_df.to_csv(metrics_path, index=False)

predictions_df = pd.concat(all_predictions, ignore_index=True)
predictions_path = os.path.join(OUTPUT_DIR, "combined_type_all_predictions.csv")
predictions_df.to_csv(predictions_path, index=False)

print("\n" + "=" * 60)
print("DONE")
print("=" * 60)

print("\nSaved:")
print(metrics_path)
print(predictions_path)

print("\nOverall fold metrics:")
print(metrics_df)