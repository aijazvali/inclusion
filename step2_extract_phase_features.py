import os
import re
import glob
import cv2
import numpy as np
import pandas as pd
from tqdm import tqdm


# ----------------------------
# CONFIG
# ----------------------------

ROOT_DIR = "."

HEAT_FOLDERS = [
    "Heat 1 images",
    "Heat 2 images",
    "Heat 3 images",
    "Heat 4 images",
]

MASK_DIR = "outputs_phase_segmentation_preview/phase_masks"

OUTPUT_CSV = "phase_features.csv"


# ----------------------------
# UTILS
# ----------------------------

def normalize_image(img):
    img = img.astype(np.float32)

    low, high = np.percentile(img, [1, 99])

    if high <= low:
        return img.astype(np.uint8)

    img = (img - low) / (high - low)
    img = np.clip(img, 0, 1)
    img = (img * 255).astype(np.uint8)

    return img


def make_save_base(image_path):
    rel_path = os.path.relpath(image_path, ROOT_DIR)
    rel_path_no_ext = os.path.splitext(rel_path)[0]

    save_base = (
        rel_path_no_ext
        .replace(os.sep, "_")
        .replace(" ", "_")
        .replace(".", "_")
    )

    return save_base


def extract_heat(image_path):
    match = re.search(r"Heat\s*(\d+)", image_path)

    if match:
        return int(match.group(1))

    return None


def extract_particle_id(image_path):
    name = os.path.splitext(os.path.basename(image_path))[0]

    try:
        return int(name)
    except:
        return name


def count_regions(binary_mask):
    binary_mask = binary_mask.astype(np.uint8)

    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(
        binary_mask,
        connectivity=8
    )

    # subtract background
    num_regions = max(num_labels - 1, 0)

    if num_regions == 0:
        return 0, 0

    areas = stats[1:, cv2.CC_STAT_AREA]
    largest_area = int(np.max(areas))

    return num_regions, largest_area


def bbox_features(particle_mask):
    ys, xs = np.where(particle_mask > 0)

    if len(xs) == 0:
        return {
            "bbox_x": 0,
            "bbox_y": 0,
            "bbox_w": 0,
            "bbox_h": 0,
            "bbox_aspect": 0,
            "bbox_extent": 0,
        }

    x_min, x_max = xs.min(), xs.max()
    y_min, y_max = ys.min(), ys.max()

    bbox_w = int(x_max - x_min + 1)
    bbox_h = int(y_max - y_min + 1)

    bbox_area = bbox_w * bbox_h
    particle_area = int(particle_mask.sum())

    bbox_aspect = bbox_w / bbox_h if bbox_h > 0 else 0
    bbox_extent = particle_area / bbox_area if bbox_area > 0 else 0

    return {
        "bbox_x": int(x_min),
        "bbox_y": int(y_min),
        "bbox_w": bbox_w,
        "bbox_h": bbox_h,
        "bbox_aspect": bbox_aspect,
        "bbox_extent": bbox_extent,
    }


# ----------------------------
# COLLECT IMAGES
# ----------------------------

all_images = []

extensions = ["*.tif", "*.tiff", "*.TIF", "*.TIFF"]

for heat_folder in HEAT_FOLDERS:
    folder_path = os.path.join(ROOT_DIR, heat_folder)

    for ext in extensions:
        files = glob.glob(
            os.path.join(folder_path, "**", ext),
            recursive=True
        )
        all_images.extend(files)

all_images = sorted(all_images)

print(f"Found {len(all_images)} raw images.")


# ----------------------------
# FEATURE EXTRACTION
# ----------------------------

rows = []
missing_masks = 0

for image_path in tqdm(all_images):
    save_base = make_save_base(image_path)
    mask_path = os.path.join(MASK_DIR, save_base + "_phase_mask.png")

    if not os.path.exists(mask_path):
        missing_masks += 1
        continue

    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    mask = cv2.imread(mask_path, cv2.IMREAD_UNCHANGED)

    if img is None or mask is None:
        continue

    img_norm = normalize_image(img)

    particle_mask = (mask > 0).astype(np.uint8)

    image_area = mask.shape[0] * mask.shape[1]
    particle_area = int(particle_mask.sum())

    row = {
        "image_path": image_path,
        "mask_path": mask_path,
        "heat": extract_heat(image_path),
        "particle_id": extract_particle_id(image_path),
        "image_h": mask.shape[0],
        "image_w": mask.shape[1],
        "image_area": image_area,
        "particle_area": particle_area,
        "particle_area_fraction": particle_area / image_area if image_area > 0 else 0,
    }

    row.update(bbox_features(particle_mask))

    if particle_area > 0:
        particle_pixels = img_norm[particle_mask > 0]

        row["particle_mean_intensity"] = float(np.mean(particle_pixels))
        row["particle_std_intensity"] = float(np.std(particle_pixels))
        row["particle_min_intensity"] = int(np.min(particle_pixels))
        row["particle_max_intensity"] = int(np.max(particle_pixels))
    else:
        row["particle_mean_intensity"] = 0
        row["particle_std_intensity"] = 0
        row["particle_min_intensity"] = 0
        row["particle_max_intensity"] = 0

    # phase-wise features
    for phase in [1, 2, 3]:
        phase_mask = (mask == phase).astype(np.uint8)
        phase_area = int(phase_mask.sum())

        row[f"phase_{phase}_area"] = phase_area
        row[f"phase_{phase}_area_fraction_image"] = (
            phase_area / image_area if image_area > 0 else 0
        )
        row[f"phase_{phase}_area_fraction_particle"] = (
            phase_area / particle_area if particle_area > 0 else 0
        )

        if phase_area > 0:
            phase_pixels = img_norm[phase_mask > 0]

            row[f"phase_{phase}_mean_intensity"] = float(np.mean(phase_pixels))
            row[f"phase_{phase}_std_intensity"] = float(np.std(phase_pixels))
            row[f"phase_{phase}_min_intensity"] = int(np.min(phase_pixels))
            row[f"phase_{phase}_max_intensity"] = int(np.max(phase_pixels))
        else:
            row[f"phase_{phase}_mean_intensity"] = 0
            row[f"phase_{phase}_std_intensity"] = 0
            row[f"phase_{phase}_min_intensity"] = 0
            row[f"phase_{phase}_max_intensity"] = 0

        num_regions, largest_region = count_regions(phase_mask)

        row[f"phase_{phase}_num_regions"] = num_regions
        row[f"phase_{phase}_largest_region_area"] = largest_region

    # overall region count
    total_regions = 0
    for phase in [1, 2, 3]:
        total_regions += row[f"phase_{phase}_num_regions"]

    row["total_phase_regions"] = total_regions

    rows.append(row)


df = pd.DataFrame(rows)
df.to_csv(OUTPUT_CSV, index=False)

print("Done.")
print(f"Saved: {OUTPUT_CSV}")
print(f"Rows saved: {len(df)}")
print(f"Missing masks: {missing_masks}")