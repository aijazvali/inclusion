import os
import copy
import glob
import warnings
warnings.filterwarnings("ignore")

import cv2
import joblib
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, f1_score, classification_report

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader


# ----------------------------
# CONFIG
# ----------------------------

CSV_CANDIDATES = [
    "phase_features_labeled.csv",
    "/kaggle/working/phase_features_labeled.csv",
    "/kaggle/working/project/phase_features_labeled.csv",
]

IMAGE_ROOT_CANDIDATES = [
    ".",
    "/kaggle/working/project",
    "/kaggle/working",
    "/kaggle/input/datasets/aijazv/project/archive",
    "/kaggle/input",
]

OUTPUT_DIR = "frontend_model"
os.makedirs(OUTPUT_DIR, exist_ok=True)

TARGET_COL = "type"
HEAT_COL = "heat"
INCLUSION_COL = "inclusion"

KEEP_CLASSES = ["oxide", "oxy-sulfide", "sulfide"]

IMG_SIZE = 128
BATCH_SIZE = 256
EPOCHS = 15
LR = 1e-3
VAL_SIZE = 0.15
RANDOM_STATE = 42
NUM_WORKERS = 2


# ----------------------------
# DEVICE
# ----------------------------

if torch.cuda.is_available():
    DEVICE = torch.device("cuda")
    USE_AMP = True
else:
    DEVICE = torch.device("cpu")
    USE_AMP = False

print("Using device:", DEVICE)


# ----------------------------
# FIND CSV
# ----------------------------

INPUT_CSV = None

for p in CSV_CANDIDATES:
    if os.path.exists(p):
        INPUT_CSV = p
        break

if INPUT_CSV is None:
    matches = glob.glob("/kaggle/**/phase_features_labeled.csv", recursive=True)
    if matches:
        INPUT_CSV = matches[0]

if INPUT_CSV is None:
    raise FileNotFoundError("Could not find phase_features_labeled.csv")

print("Using CSV:", INPUT_CSV)


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
        return np.clip(img / 255.0, 0, 1)

    img = (img - low) / (high - low)
    img = np.clip(img, 0, 1)

    return img


def resolve_image_path(old_path):
    if isinstance(old_path, str) and os.path.exists(old_path):
        return old_path

    old_path = str(old_path)

    marker = None

    for h in ["Heat 1 images", "Heat 2 images", "Heat 3 images", "Heat 4 images"]:
        if h in old_path:
            marker = h
            break

    if marker is None:
        return None

    suffix = old_path[old_path.index(marker):].replace("\\", "/")

    for root in IMAGE_ROOT_CANDIDATES:
        candidate = os.path.join(root, suffix)
        if os.path.exists(candidate):
            return candidate

    return None


def read_image(path):
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)

    if img is None:
        raise ValueError(f"Could not read image: {path}")

    if img.shape != (IMG_SIZE, IMG_SIZE):
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

    allowed_prefixes = ["phase_1_", "phase_2_", "phase_3_"]

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


# ----------------------------
# DATASET
# ----------------------------

class InclusionCombinedDataset(Dataset):
    def __init__(self, image_paths, tab_features, labels, train=False):
        self.image_paths = image_paths
        self.tab_features = tab_features.astype(np.float32)
        self.labels = np.array(labels, dtype=np.int64)
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

        img = torch.from_numpy(img).float().unsqueeze(0)
        tab = torch.from_numpy(self.tab_features[idx]).float()
        label = torch.tensor(self.labels[idx], dtype=torch.long)

        return img, tab, label


# ----------------------------
# MODEL
# ----------------------------

class CombinedCNNTabular(nn.Module):
    def __init__(self, num_tab_features, num_classes):
        super().__init__()

        self.image_encoder = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),

            nn.Conv2d(32, 64, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),

            nn.Conv2d(64, 128, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),

            nn.Conv2d(128, 256, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),

            nn.Conv2d(256, 384, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(384),
            nn.ReLU(inplace=True),

            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
        )

        self.tabular_encoder = nn.Sequential(
            nn.Linear(num_tab_features, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.20),

            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(inplace=True),
        )

        self.classifier = nn.Sequential(
            nn.Linear(384 + 64, 192),
            nn.ReLU(inplace=True),
            nn.Dropout(0.35),
            nn.Linear(192, num_classes),
        )

    def forward(self, img, tab):
        img_feat = self.image_encoder(img)
        tab_feat = self.tabular_encoder(tab)
        x = torch.cat([img_feat, tab_feat], dim=1)
        return self.classifier(x)


# ----------------------------
# TRAIN / EVAL
# ----------------------------

def train_one_epoch(model, loader, criterion, optimizer, scaler_amp):
    model.train()

    total_loss = 0
    all_true = []
    all_pred = []

    for imgs, tabs, labels in loader:
        imgs = imgs.to(DEVICE)
        tabs = tabs.to(DEVICE)
        labels = labels.to(DEVICE)

        optimizer.zero_grad(set_to_none=True)

        if USE_AMP:
            with torch.cuda.amp.autocast():
                logits = model(imgs, tabs)
                loss = criterion(logits, labels)

            scaler_amp.scale(loss).backward()
            scaler_amp.step(optimizer)
            scaler_amp.update()
        else:
            logits = model(imgs, tabs)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()

        total_loss += loss.item() * imgs.size(0)

        preds = torch.argmax(logits, dim=1)

        all_true.extend(labels.detach().cpu().tolist())
        all_pred.extend(preds.detach().cpu().tolist())

    avg_loss = total_loss / len(loader.dataset)
    acc = accuracy_score(all_true, all_pred)

    return avg_loss, acc


@torch.no_grad()
def evaluate(model, loader, criterion):
    model.eval()

    total_loss = 0
    all_true = []
    all_pred = []

    for imgs, tabs, labels in loader:
        imgs = imgs.to(DEVICE)
        tabs = tabs.to(DEVICE)
        labels = labels.to(DEVICE)

        if USE_AMP:
            with torch.cuda.amp.autocast():
                logits = model(imgs, tabs)
                loss = criterion(logits, labels)
        else:
            logits = model(imgs, tabs)
            loss = criterion(logits, labels)

        total_loss += loss.item() * imgs.size(0)

        preds = torch.argmax(logits, dim=1)

        all_true.extend(labels.detach().cpu().tolist())
        all_pred.extend(preds.detach().cpu().tolist())

    avg_loss = total_loss / len(loader.dataset)

    return avg_loss, np.array(all_true), np.array(all_pred)


# ----------------------------
# LOAD DATA
# ----------------------------

df = pd.read_csv(INPUT_CSV)

df[TARGET_COL] = clean_label_series(df[TARGET_COL])
df = df[df[TARGET_COL].isin(KEEP_CLASSES)].copy()

if INCLUSION_COL in df.columns:
    df[INCLUSION_COL] = pd.to_numeric(df[INCLUSION_COL], errors="coerce")
    df = df[df[INCLUSION_COL] == 1].copy()

df["resolved_image_path"] = df["image_path"].apply(resolve_image_path)
df = df.dropna(subset=["resolved_image_path"]).copy()

feature_cols = get_feature_columns(df)

classes = sorted(df[TARGET_COL].unique())
label_to_idx = {c: i for i, c in enumerate(classes)}
idx_to_label = {i: c for c, i in label_to_idx.items()}

print("Classes:", classes)
print(df[TARGET_COL].value_counts())
print("Feature count:", len(feature_cols))
print("Rows:", len(df))


# ----------------------------
# SPLIT
# ----------------------------

train_df, val_df = train_test_split(
    df,
    test_size=VAL_SIZE,
    random_state=RANDOM_STATE,
    stratify=df[TARGET_COL],
)

imputer = SimpleImputer(strategy="median")
scaler_tab = StandardScaler()

X_train_tab = imputer.fit_transform(train_df[feature_cols])
X_val_tab = imputer.transform(val_df[feature_cols])

X_train_tab = scaler_tab.fit_transform(X_train_tab)
X_val_tab = scaler_tab.transform(X_val_tab)

y_train = train_df[TARGET_COL].map(label_to_idx).astype(int).values
y_val = val_df[TARGET_COL].map(label_to_idx).astype(int).values

train_dataset = InclusionCombinedDataset(
    train_df["resolved_image_path"].tolist(),
    X_train_tab,
    y_train,
    train=True,
)

val_dataset = InclusionCombinedDataset(
    val_df["resolved_image_path"].tolist(),
    X_val_tab,
    y_val,
    train=False,
)

train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=NUM_WORKERS,
    pin_memory=torch.cuda.is_available(),
)

val_loader = DataLoader(
    val_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=NUM_WORKERS,
    pin_memory=torch.cuda.is_available(),
)

model = CombinedCNNTabular(
    num_tab_features=len(feature_cols),
    num_classes=len(classes),
).to(DEVICE)

train_counts = train_df[TARGET_COL].value_counts()
class_weights = []

for c in classes:
    class_weights.append(len(train_df) / (len(classes) * train_counts[c]))

class_weights = torch.tensor(class_weights, dtype=torch.float32).to(DEVICE)

criterion = nn.CrossEntropyLoss(weight=class_weights)
optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)
scaler_amp = torch.cuda.amp.GradScaler(enabled=USE_AMP)

best_val_loss = float("inf")
best_state = None

for epoch in range(1, EPOCHS + 1):
    train_loss, train_acc = train_one_epoch(
        model,
        train_loader,
        criterion,
        optimizer,
        scaler_amp,
    )

    val_loss, y_true, y_pred = evaluate(model, val_loader, criterion)

    val_acc = accuracy_score(y_true, y_pred)
    val_macro_f1 = f1_score(y_true, y_pred, average="macro")

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

val_loss, y_true, y_pred = evaluate(model, val_loader, criterion)

print("\nFinal validation report:")
print(classification_report(
    [idx_to_label[i] for i in y_true],
    [idx_to_label[i] for i in y_pred],
    zero_division=0,
))

checkpoint_path = os.path.join(OUTPUT_DIR, "combined_3class_frontend_model.pt")
preprocess_path = os.path.join(OUTPUT_DIR, "combined_3class_preprocess.joblib")

torch.save(
    {
        "model_state_dict": model.state_dict(),
        "classes": classes,
        "feature_cols": feature_cols,
        "label_to_idx": label_to_idx,
        "idx_to_label": idx_to_label,
        "img_size": IMG_SIZE,
    },
    checkpoint_path,
)

joblib.dump(
    {
        "imputer": imputer,
        "scaler": scaler_tab,
        "feature_cols": feature_cols,
        "classes": classes,
    },
    preprocess_path,
)

print("\nSaved:")
print(checkpoint_path)
print(preprocess_path)