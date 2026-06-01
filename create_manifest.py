import pandas as pd
from pathlib import Path

excel_path = "2018_AlCaSMn_20kV_4.xlsx"

heat_image_dirs = {
    1: "Heat 1 images/imgs",
    2: "Heat 2 images/imgs",
    3: "Heat 3 images/imgs",
    4: "Heat 4 images/imgs",
}

df = pd.read_excel(excel_path)

manifest_rows = []

for _, row in df.iterrows():
    heat = int(row["Heat"])
    part = int(row["Part"])
    label = int(row["inclusion"])

    image_name = f"{part:05d}.TIF"
    image_path = Path(heat_image_dirs[heat]) / image_name

    sample_id = f"{heat}-{part}"

    manifest_rows.append({
        "sample_id": sample_id,
        "heat": heat,
        "part": part,
        "image_path": str(image_path),
        "label": label,
        "label_name": "inclusion" if label == 1 else "non-inclusion"
    })

manifest = pd.DataFrame(manifest_rows)

manifest.to_csv("inclusion_binary_manifest.csv", index=False)

print("Manifest created.")
print("Total samples:", len(manifest))
print(manifest["label_name"].value_counts())
print(manifest.head())