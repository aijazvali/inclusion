import pandas as pd

df = pd.read_csv("inclusion_binary_manifest_with_splits.csv")

sample_ids = [
    "3-5347",
    "1-12364",
    "1-11134",
    "2-1343",
    "4-1162",
    "1-19419",
    "1-13628",
    "3-12805",
    "4-1939",
    "1-16190",
    "4-217",
    "1-11209",
    "3-3024",
    "4-3254",
    "3-4848",
    "3-2437",
]

out = df[df["sample_id"].isin(sample_ids)][
    ["sample_id", "heat", "part", "label", "label_name", "image_path"]
].sort_values("sample_id")

print(out.to_string(index=False))