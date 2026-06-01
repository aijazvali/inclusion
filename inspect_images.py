import pandas as pd
import numpy as np
from pathlib import Path
from PIL import Image, ImageOps, ImageDraw

manifest_path = "inclusion_binary_manifest_with_splits.csv"

df = pd.read_csv(manifest_path)

print("Total samples:", len(df))

# --------------------------------------------------
# 1. Inspect image properties
# --------------------------------------------------

records = []

for idx, row in df.iterrows():
    image_path = Path(row["image_path"])

    try:
        img = Image.open(image_path)
        arr = np.array(img)

        records.append({
            "sample_id": row["sample_id"],
            "label": row["label"],
            "label_name": row["label_name"],
            "image_path": row["image_path"],
            "mode": img.mode,
            "width": img.width,
            "height": img.height,
            "array_shape": str(arr.shape),
            "dtype": str(arr.dtype),
            "min_pixel": float(arr.min()),
            "max_pixel": float(arr.max()),
            "mean_pixel": float(arr.mean()),
            "std_pixel": float(arr.std()),
        })

    except Exception as e:
        records.append({
            "sample_id": row["sample_id"],
            "label": row["label"],
            "label_name": row["label_name"],
            "image_path": row["image_path"],
            "error": str(e),
        })

report = pd.DataFrame(records)

report.to_csv("image_inspection_report.csv", index=False)

print("\nSaved: image_inspection_report.csv")

# --------------------------------------------------
# 2. Print summary
# --------------------------------------------------

if "error" in report.columns:
    errors = report[report["error"].notna()]
    print("Image loading errors:", len(errors))
    if len(errors) > 0:
        print(errors.head())
else:
    print("Image loading errors: 0")

print("\nImage modes:")
print(report["mode"].value_counts())

print("\nImage sizes:")
print(report[["width", "height"]].value_counts().head(20))

print("\nPixel dtype:")
print(report["dtype"].value_counts())

print("\nPixel range summary:")
print(report[["min_pixel", "max_pixel", "mean_pixel", "std_pixel"]].describe())

# --------------------------------------------------
# 3. Create visual contact sheet
# --------------------------------------------------

sample_df = (
    df.groupby("label_name", group_keys=False)
    .apply(lambda x: x.sample(n=8, random_state=42))
    .reset_index(drop=True)
)

thumb_size = 160
label_height = 30
cols = 4
rows = int(np.ceil(len(sample_df) / cols))

sheet = Image.new(
    "RGB",
    (cols * thumb_size, rows * (thumb_size + label_height)),
    color="white"
)

draw = ImageDraw.Draw(sheet)

for i, row in sample_df.iterrows():
    img = Image.open(row["image_path"]).convert("L")

    # Resize while keeping aspect ratio, then pad to square
    img = ImageOps.pad(img, (thumb_size, thumb_size), color=0)
    img = img.convert("RGB")

    x = (i % cols) * thumb_size
    y = (i // cols) * (thumb_size + label_height)

    sheet.paste(img, (x, y))

    label_text = f'{row["sample_id"]} | {row["label_name"]}'
    draw.text((x + 5, y + thumb_size + 5), label_text, fill="black")

sheet.save("sample_contact_sheet.png")

print("\nSaved: sample_contact_sheet.png")
