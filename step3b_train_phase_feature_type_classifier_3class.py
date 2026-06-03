import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    f1_score,
    classification_report,
    confusion_matrix,
)
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline


# ----------------------------
# CONFIG
# ----------------------------

INPUT_CSV = "phase_features_labeled.csv"
OUTPUT_DIR = "outputs_phase_feature_type_classifier_3class"

TARGET_COL = "type"
DROP_CLASSES = ["other", "nitride", "non-inclusion", "non-inc", "non inclusion"]

# Since binary classification is already done, we train type classifier only on inclusions.
FILTER_INCLUSION_ONLY = True

# Remove extremely tiny classes such as nitride if only 1-2 samples exist.
MIN_CLASS_COUNT = 10

os.makedirs(OUTPUT_DIR, exist_ok=True)


# ----------------------------
# HELPERS
# ----------------------------

def find_col(df, candidates):
    for c in candidates:
        if c in df.columns:
            return c
    return None


def clean_label_series(s):
    return (
        s.astype(str)
        .str.strip()
        .str.replace("_", "-", regex=False)
        .str.lower()
    )


def get_feature_columns(df):
    """
    Use only features generated from our segmentation step.
    Avoid EDS columns, original Excel morphology columns, labels, paths, etc.
    """

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

    # keep only numeric columns
    numeric_feature_cols = []

    for col in feature_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")
        if pd.api.types.is_numeric_dtype(df[col]):
            numeric_feature_cols.append(col)

    return numeric_feature_cols


def plot_confusion_matrix(cm, labels, save_path, title):
    fig, ax = plt.subplots(figsize=(8, 7))

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
            ax.text(
                j,
                i,
                str(cm[i, j]),
                ha="center",
                va="center",
            )

    fig.colorbar(im, ax=ax)
    plt.tight_layout()
    plt.savefig(save_path, dpi=200)
    plt.close()


# ----------------------------
# LOAD DATA
# ----------------------------

df = pd.read_csv(INPUT_CSV)

print("Loaded:", INPUT_CSV)
print("Shape:", df.shape)

heat_col = find_col(df, ["heat", "Heat", "HEAT"])
inclusion_col = find_col(df, ["inclusion", "Inclusion"])
target_col = find_col(df, [TARGET_COL])

if heat_col is None:
    raise ValueError("Could not find heat column. Expected one of: heat, Heat, HEAT")

if target_col is None:
    raise ValueError(f"Could not find target column: {TARGET_COL}")

df[heat_col] = pd.to_numeric(df[heat_col], errors="coerce")
df = df.dropna(subset=[heat_col])
df[heat_col] = df[heat_col].astype(int)

df[target_col] = clean_label_series(df[target_col])
df = df[~df[target_col].isin(["nan", "", "none"])]
df = df[~df[target_col].isin(DROP_CLASSES)].copy()

if FILTER_INCLUSION_ONLY:
    if inclusion_col is not None:
        df[inclusion_col] = pd.to_numeric(df[inclusion_col], errors="coerce")
        df = df[df[inclusion_col] == 1].copy()
    else:
        # fallback: remove non-inclusion labels
        df = df[~df[target_col].isin(["non-inclusion", "non-inc", "non inclusion"])].copy()

# remove rare classes
class_counts = df[target_col].value_counts()
keep_classes = class_counts[class_counts >= MIN_CLASS_COUNT].index.tolist()

df = df[df[target_col].isin(keep_classes)].copy()

print("\nClass counts after filtering:")
print(df[target_col].value_counts())

feature_cols = get_feature_columns(df)

print("\nNumber of features:", len(feature_cols))
print("Features:")
for c in feature_cols:
    print(" -", c)

if len(feature_cols) == 0:
    raise ValueError("No segmentation feature columns found.")

heats = sorted(df[heat_col].unique())

print("\nHeat-wise folds:", heats)


# ----------------------------
# TRAIN HEAT-WISE FOLDS
# ----------------------------

all_metrics = []
all_predictions = []
all_importances = []

for test_heat in heats:
    print("\n" + "=" * 60)
    print(f"Fold: Test Heat {test_heat}")
    print("=" * 60)

    train_df = df[df[heat_col] != test_heat].copy()
    test_df = df[df[heat_col] == test_heat].copy()

    X_train = train_df[feature_cols]
    y_train = train_df[target_col]

    X_test = test_df[feature_cols]
    y_test = test_df[target_col]

    print("Train size:", len(train_df))
    print("Test size:", len(test_df))

    print("\nTrain class counts:")
    print(y_train.value_counts())

    print("\nTest class counts:")
    print(y_test.value_counts())

    model = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            (
                "clf",
                ExtraTreesClassifier(
                    n_estimators=500,
                    random_state=42,
                    class_weight="balanced",
                    n_jobs=-1,
                    max_features="sqrt",
                ),
            ),
        ]
    )

    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    labels = sorted(y_train.unique())

    acc = accuracy_score(y_test, y_pred)
    bal_acc = balanced_accuracy_score(y_test, y_pred)
    macro_f1 = f1_score(y_test, y_pred, average="macro")
    weighted_f1 = f1_score(y_test, y_pred, average="weighted")

    metrics_row = {
        "test_heat": test_heat,
        "train_size": len(train_df),
        "test_size": len(test_df),
        "accuracy": acc,
        "balanced_accuracy": bal_acc,
        "macro_f1": macro_f1,
        "weighted_f1": weighted_f1,
    }

    all_metrics.append(metrics_row)

    print("\nMetrics:")
    for k, v in metrics_row.items():
        print(f"{k}: {v}")

    report = classification_report(
        y_test,
        y_pred,
        labels=labels,
        zero_division=0,
    )

    report_path = os.path.join(
        OUTPUT_DIR,
        f"classification_report_type_test_heat_{test_heat}.txt"
    )

    with open(report_path, "w") as f:
        f.write(report)

    print("\nClassification report:")
    print(report)

    cm = confusion_matrix(y_test, y_pred, labels=labels)

    cm_path = os.path.join(
        OUTPUT_DIR,
        f"confusion_matrix_type_test_heat_{test_heat}.png"
    )

    plot_confusion_matrix(
        cm,
        labels,
        cm_path,
        title=f"Type Classification - Test Heat {test_heat}",
    )

    # predictions
    pred_df = test_df[["image_path", "mask_path", heat_col, "particle_id", target_col]].copy()
    pred_df = pred_df.rename(columns={target_col: "true_type"})
    pred_df["pred_type"] = y_pred
    pred_df["test_heat"] = test_heat

    all_predictions.append(pred_df)

    # feature importances
    clf = model.named_steps["clf"]

    fold_importance = pd.DataFrame(
        {
            "feature": feature_cols,
            "importance": clf.feature_importances_,
            "test_heat": test_heat,
        }
    )

    all_importances.append(fold_importance)


# ----------------------------
# SAVE OVERALL OUTPUTS
# ----------------------------

metrics_df = pd.DataFrame(all_metrics)
metrics_path = os.path.join(OUTPUT_DIR, "phase_feature_type_fold_metrics.csv")
metrics_df.to_csv(metrics_path, index=False)

predictions_df = pd.concat(all_predictions, ignore_index=True)
predictions_path = os.path.join(OUTPUT_DIR, "phase_feature_type_all_predictions.csv")
predictions_df.to_csv(predictions_path, index=False)

importance_df = pd.concat(all_importances, ignore_index=True)
importance_path = os.path.join(OUTPUT_DIR, "phase_feature_type_feature_importances_all_folds.csv")
importance_df.to_csv(importance_path, index=False)

mean_importance_df = (
    importance_df
    .groupby("feature", as_index=False)["importance"]
    .mean()
    .sort_values("importance", ascending=False)
)

mean_importance_path = os.path.join(OUTPUT_DIR, "phase_feature_type_feature_importances_mean.csv")
mean_importance_df.to_csv(mean_importance_path, index=False)

print("\n" + "=" * 60)
print("DONE")
print("=" * 60)

print("\nSaved:")
print(metrics_path)
print(predictions_path)
print(importance_path)
print(mean_importance_path)

print("\nOverall fold metrics:")
print(metrics_df)

print("\nTop 15 segmentation features:")
print(mean_importance_df.head(15))