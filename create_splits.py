import pandas as pd
from pathlib import Path

manifest_path = "inclusion_binary_manifest.csv"

df = pd.read_csv(manifest_path)

# --------------------------------------------------
# 1. Check whether image paths actually exist
# --------------------------------------------------

df["image_exists"] = df["image_path"].apply(lambda p: Path(p).exists())

missing = df[df["image_exists"] == False]

print("Total samples:", len(df))
print("Images found:", df["image_exists"].sum())
print("Missing images:", len(missing))

if len(missing) > 0:
    print("\nSome missing image examples:")
    print(missing[["sample_id", "image_path"]].head(20))
    raise FileNotFoundError("Some images are missing. Fix paths before continuing.")

# --------------------------------------------------
# 2. Create heat-wise folds
# --------------------------------------------------
# Meaning:
# train = used to train model
# val   = used to tune/check model during training
# test  = final unseen heat for evaluation

fold_configs = {
    "fold_test_heat_1": {
        "test": [1],
        "val": [2],
        "train": [3, 4],
    },
    "fold_test_heat_2": {
        "test": [2],
        "val": [1],
        "train": [3, 4],
    },
    "fold_test_heat_3": {
        "test": [3],
        "val": [4],
        "train": [1, 2],
    },
    "fold_test_heat_4": {
        "test": [4],
        "val": [3],
        "train": [1, 2],
    },
}

for fold_name, config in fold_configs.items():
    df[fold_name] = "unused"

    df.loc[df["heat"].isin(config["train"]), fold_name] = "train"
    df.loc[df["heat"].isin(config["val"]), fold_name] = "val"
    df.loc[df["heat"].isin(config["test"]), fold_name] = "test"

# --------------------------------------------------
# 3. Save updated manifest
# --------------------------------------------------

output_path = "inclusion_binary_manifest_with_splits.csv"
df.to_csv(output_path, index=False)

print("\nSaved:", output_path)

# --------------------------------------------------
# 4. Print split summary
# --------------------------------------------------

for fold_name in fold_configs.keys():
    print("\n" + "=" * 60)
    print(fold_name)
    print("=" * 60)

    print(df[fold_name].value_counts())

    print("\nClass balance:")
    print(pd.crosstab(df[fold_name], df["label_name"]))