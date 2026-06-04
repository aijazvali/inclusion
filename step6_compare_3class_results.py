import os
import pandas as pd
import matplotlib.pyplot as plt

OUTPUT_DIR = "outputs_3class_model_comparison"
os.makedirs(OUTPUT_DIR, exist_ok=True)

files = {
    "phase_features_only": "outputs_phase_feature_type_classifier_3class/phase_feature_type_fold_metrics.csv",
    "image_only_cnn": "outputs_image_only_type_classifier_3class/image_only_type_fold_metrics.csv",
    "image_plus_phase_features": "outputs_image_plus_phase_features_3class/combined_type_fold_metrics.csv",
}

rows = []

for model_name, path in files.items():
    df = pd.read_csv(path)

    row = {
        "model": model_name,
        "mean_accuracy": df["accuracy"].mean(),
        "std_accuracy": df["accuracy"].std(),
        "mean_balanced_accuracy": df["balanced_accuracy"].mean(),
        "std_balanced_accuracy": df["balanced_accuracy"].std(),
        "mean_macro_f1": df["macro_f1"].mean(),
        "std_macro_f1": df["macro_f1"].std(),
        "mean_weighted_f1": df["weighted_f1"].mean(),
        "std_weighted_f1": df["weighted_f1"].std(),
    }

    rows.append(row)

summary = pd.DataFrame(rows)
summary = summary.sort_values("mean_macro_f1", ascending=False)

summary_path = os.path.join(OUTPUT_DIR, "model_comparison_summary.csv")
summary.to_csv(summary_path, index=False)

print(summary)

# Save markdown table for report
markdown_path = os.path.join(OUTPUT_DIR, "model_comparison_summary.md")

with open(markdown_path, "w") as f:
    f.write(summary.to_markdown(index=False))

# Plot macro F1 comparison
plt.figure(figsize=(8, 5))
plt.bar(summary["model"], summary["mean_macro_f1"])
plt.ylabel("Mean Macro F1")
plt.xlabel("Model")
plt.title("3-Class Inclusion Type Classification")
plt.xticks(rotation=30, ha="right")
plt.tight_layout()

plot_path = os.path.join(OUTPUT_DIR, "macro_f1_comparison.png")
plt.savefig(plot_path, dpi=200)
plt.close()

# Plot balanced accuracy comparison
plt.figure(figsize=(8, 5))
plt.bar(summary["model"], summary["mean_balanced_accuracy"])
plt.ylabel("Mean Balanced Accuracy")
plt.xlabel("Model")
plt.title("3-Class Inclusion Type Classification")
plt.xticks(rotation=30, ha="right")
plt.tight_layout()

plot_path2 = os.path.join(OUTPUT_DIR, "balanced_accuracy_comparison.png")
plt.savefig(plot_path2, dpi=200)
plt.close()

print("\nSaved:")
print(summary_path)
print(markdown_path)
print(plot_path)
print(plot_path2)