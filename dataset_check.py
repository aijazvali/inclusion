import pandas as pd
from pathlib import Path
from PIL import Image

import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms


# --------------------------------------------------
# 1. Configuration
# --------------------------------------------------

manifest_path = "inclusion_binary_manifest_with_splits.csv"

fold_column = "fold_test_heat_1"
batch_size = 32


# --------------------------------------------------
# 2. Custom PyTorch Dataset
# --------------------------------------------------

class InclusionDataset(Dataset):
    def __init__(self, dataframe, transform=None):
        self.df = dataframe.reset_index(drop=True)
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]

        image_path = row["image_path"]
        label = int(row["label"])

        image = Image.open(image_path).convert("L")

        if self.transform is not None:
            image = self.transform(image)

        label = torch.tensor(label, dtype=torch.long)

        return image, label


# --------------------------------------------------
# 3. Load manifest and select fold
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
# 4. Basic transform: convert image to tensor
# --------------------------------------------------

basic_transform = transforms.Compose([
    transforms.ToTensor()
])

train_dataset_basic = InclusionDataset(train_df, transform=basic_transform)

train_loader_basic = DataLoader(
    train_dataset_basic,
    batch_size=batch_size,
    shuffle=False,
    num_workers=0
)


# --------------------------------------------------
# 5. Compute training mean and std
# --------------------------------------------------

def compute_mean_std(loader):
    total_sum = 0.0
    total_squared_sum = 0.0
    total_pixels = 0

    for images, _ in loader:
        # images shape: [batch, 1, 128, 128]
        total_sum += images.sum().item()
        total_squared_sum += (images ** 2).sum().item()
        total_pixels += images.numel()

    mean = total_sum / total_pixels
    variance = (total_squared_sum / total_pixels) - (mean ** 2)
    std = variance ** 0.5

    return mean, std


mean, std = compute_mean_std(train_loader_basic)

print("\nTraining-set pixel statistics:")
print("Mean:", mean)
print("Std:", std)


# --------------------------------------------------
# 6. Final preprocessing transform
# --------------------------------------------------

final_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[mean], std=[std])
])


train_dataset = InclusionDataset(train_df, transform=final_transform)
val_dataset = InclusionDataset(val_df, transform=final_transform)
test_dataset = InclusionDataset(test_df, transform=final_transform)


train_loader = DataLoader(
    train_dataset,
    batch_size=batch_size,
    shuffle=True,
    num_workers=0
)

val_loader = DataLoader(
    val_dataset,
    batch_size=batch_size,
    shuffle=False,
    num_workers=0
)

test_loader = DataLoader(
    test_dataset,
    batch_size=batch_size,
    shuffle=False,
    num_workers=0
)


# --------------------------------------------------
# 7. Sanity check one batch
# --------------------------------------------------

images, labels = next(iter(train_loader))

print("\nOne training batch:")
print("Images shape:", images.shape)
print("Labels shape:", labels.shape)
print("Image tensor dtype:", images.dtype)
print("Label tensor dtype:", labels.dtype)
print("Min pixel after normalization:", images.min().item())
print("Max pixel after normalization:", images.max().item())
print("Labels:", labels[:20].tolist())


# --------------------------------------------------
# 8. Final confirmation
# --------------------------------------------------

assert images.shape[1] == 1
assert images.shape[2] == 128
assert images.shape[3] == 128
assert labels.ndim == 1

print("\nDataset check passed.")