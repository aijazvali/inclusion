import numpy as np
import pandas as pd
from pathlib import Path
from PIL import Image, ImageOps, ImageDraw

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

        return image, label, idx


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
# 5. Load checkpoint
# --------------------------------------------------

checkpoint = torch.load(checkpoint_path, map_location=device)

mean = checkpoint["mean"]
std = checkpoint["std"]

print("Loaded checkpoint:", checkpoint_path)
print("Best epoch:", checkpoint["epoch"])
print("Best val balanced accuracy:", checkpoint["best_val_balanced_accuracy"])
print("Mean:", mean)
print("Std:", std)


# --------------------------------------------------
# 6. Load test data
# --------------------------------------------------

df = pd.read_csv(manifest_path)

test_df = df[df[fold_column] == "test"].copy().reset_index(drop=True)

print("Test samples:", len(test_df))
print(test_df["label_name"].value_counts())


transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[mean], std=[std]),
])

test_dataset = InclusionDataset(test_df, transform=transform)

test_loader = DataLoader(
    test_dataset,
    batch_size=batch_size,
    shuffle=False,
    num_workers=0,
)


# --------------------------------------------------
# 7. Predict on test set
# --------------------------------------------------

model = InclusionCNN(num_classes=2).to(device)
model.load_state_dict(checkpoint["model_state_dict"])
model.eval()

all_rows = []

with torch.no_grad():
    for images, labels, indices in test_loader:
        images = images.to(device)
        labels = labels.to(device)

        outputs = model(images)
        probs = torch.softmax(outputs, dim=1)

        pred_labels = probs.argmax(dim=1)

        for i in range(len(labels)):
            idx = int(indices[i].item())

            true_label = int(labels[i].item())
            pred_label = int(pred_labels[i].item())

            prob_non_inclusion = float(probs[i, 0].item())
            prob_inclusion = float(probs[i, 1].item())

            confidence = max(prob_non_inclusion, prob_inclusion)

            row = test_df.iloc[idx].to_dict()

            row["true_label"] = true_label
            row["true_label_name"] = "inclusion" if true_label == 1 else "non-inclusion"

            row["pred_label"] = pred_label
            row["pred_label_name"] = "inclusion" if pred_label == 1 else "non-inclusion"

            row["prob_non_inclusion"] = prob_non_inclusion
            row["prob_inclusion"] = prob_inclusion
            row["confidence"] = confidence
            row["correct"] = true_label == pred_label

            if true_label == 0 and pred_label == 1:
                row["error_type"] = "false_positive"
            elif true_label == 1 and pred_label == 0:
                row["error_type"] = "false_negative"
            else:
                row["error_type"] = "correct"

            all_rows.append(row)

pred_df = pd.DataFrame(all_rows)

pred_df.to_csv("predictions_fold1_test.csv", index=False)

print("\nSaved: predictions_fold1_test.csv")


# --------------------------------------------------
# 8. Confusion matrix
# --------------------------------------------------

tp = ((pred_df["true_label"] == 1) & (pred_df["pred_label"] == 1)).sum()
tn = ((pred_df["true_label"] == 0) & (pred_df["pred_label"] == 0)).sum()
fp = ((pred_df["true_label"] == 0) & (pred_df["pred_label"] == 1)).sum()
fn = ((pred_df["true_label"] == 1) & (pred_df["pred_label"] == 0)).sum()

confusion = pd.DataFrame({
    "pred_non_inclusion": [tn, fn],
    "pred_inclusion": [fp, tp],
}, index=["actual_non_inclusion", "actual_inclusion"])

confusion.to_csv("confusion_matrix_fold1_test.csv")

print("\nConfusion matrix:")
print(confusion)

print("\nError counts:")
print(pred_df["error_type"].value_counts())


# --------------------------------------------------
# 9. Contact sheet function
# --------------------------------------------------

def make_contact_sheet(dataframe, output_path, title, n=32):
    if len(dataframe) == 0:
        print(f"No images for {output_path}")
        return

    show_df = dataframe.head(n).copy().reset_index(drop=True)

    thumb_size = 160
    label_height = 45
    cols = 4
    rows = int(np.ceil(len(show_df) / cols))

    sheet_width = cols * thumb_size
    sheet_height = rows * (thumb_size + label_height) + 40

    sheet = Image.new("RGB", (sheet_width, sheet_height), color="white")
    draw = ImageDraw.Draw(sheet)

    draw.text((10, 10), title, fill="black")

    for i, row in show_df.iterrows():
        img = Image.open(row["image_path"]).convert("L")
        img = ImageOps.pad(img, (thumb_size, thumb_size), color=0)
        img = img.convert("RGB")

        x = (i % cols) * thumb_size
        y = 40 + (i // cols) * (thumb_size + label_height)

        sheet.paste(img, (x, y))

        label_text_1 = f'{row["sample_id"]}'
        label_text_2 = f'T:{row["true_label_name"]} P:{row["pred_label_name"]}'
        label_text_3 = f'Pincl:{row["prob_inclusion"]:.2f} Conf:{row["confidence"]:.2f}'

        draw.text((x + 5, y + thumb_size + 2), label_text_1, fill="black")
        draw.text((x + 5, y + thumb_size + 16), label_text_2, fill="black")
        draw.text((x + 5, y + thumb_size + 30), label_text_3, fill="black")

    sheet.save(output_path)
    print("Saved:", output_path)


# --------------------------------------------------
# 10. Save error visualizations
# --------------------------------------------------

false_positives = pred_df[pred_df["error_type"] == "false_positive"].copy()
false_negatives = pred_df[pred_df["error_type"] == "false_negative"].copy()

# Highest confidence mistakes first
false_positives = false_positives.sort_values("confidence", ascending=False)
false_negatives = false_negatives.sort_values("confidence", ascending=False)

make_contact_sheet(
    false_positives,
    "false_positives_fold1.png",
    "False positives: true non-inclusion, predicted inclusion",
    n=32,
)

make_contact_sheet(
    false_negatives,
    "false_negatives_fold1.png",
    "False negatives: true inclusion, predicted non-inclusion",
    n=32,
)

# Low-confidence errors: model was unsure and wrong
errors = pred_df[pred_df["correct"] == False].copy()
errors = errors.sort_values("confidence", ascending=True)

make_contact_sheet(
    errors,
    "low_confidence_errors_fold1.png",
    "Low-confidence wrong predictions",
    n=32,
)


# --------------------------------------------------
# 11. Print useful summaries
# --------------------------------------------------

print("\nTop 10 highest-confidence false positives:")
print(false_positives[
    ["sample_id", "true_label_name", "pred_label_name", "prob_inclusion", "confidence", "image_path"]
].head(10).to_string(index=False))

print("\nTop 10 highest-confidence false negatives:")
print(false_negatives[
    ["sample_id", "true_label_name", "pred_label_name", "prob_inclusion", "confidence", "image_path"]
].head(10).to_string(index=False))

print("\nStep 8 error analysis complete.")