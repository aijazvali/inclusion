import os
import glob
import cv2
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from tqdm import tqdm


# ----------------------------
# CONFIG
# ----------------------------

ROOT_DIR = "."   # change this if your Heat folders are somewhere else

HEAT_FOLDERS = [
    "Heat 1 images",
    "Heat 2 images",
    "Heat 3 images",
    "Heat 4 images",
]

OUTPUT_DIR = "outputs_phase_segmentation_preview"
MASK_DIR = os.path.join(OUTPUT_DIR, "phase_masks")
PREVIEW_DIR = os.path.join(OUTPUT_DIR, "previews")

NUM_PREVIEWS = 50   # first inspect only 50 images

os.makedirs(MASK_DIR, exist_ok=True)
os.makedirs(PREVIEW_DIR, exist_ok=True)


# ----------------------------
# IMAGE UTILS
# ----------------------------

def read_grayscale(path):
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)

    if img is None:
        raise ValueError(f"Could not read image: {path}")

    return img


def normalize_image(img):
    """
    Robust contrast normalization using percentile clipping.
    """
    img = img.astype(np.float32)

    low, high = np.percentile(img, [1, 99])

    if high <= low:
        return img.astype(np.uint8)

    img = (img - low) / (high - low)
    img = np.clip(img, 0, 1)
    img = (img * 255).astype(np.uint8)

    return img


def clean_mask(mask):
    """
    Morphological cleanup.
    """
    mask = mask.astype(np.uint8)

    kernel = np.ones((3, 3), np.uint8)

    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)

    return mask


def keep_best_component(mask):
    """
    Keep the connected component most likely to be the main particle.
    Preference:
    - large area
    - near image center
    - not mostly border artifact
    """
    h, w = mask.shape

    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(
        mask.astype(np.uint8),
        connectivity=8
    )

    if num_labels <= 1:
        return mask.astype(np.uint8)

    image_center = np.array([w / 2, h / 2])
    best_label = None
    best_score = -1

    for label in range(1, num_labels):
        area = stats[label, cv2.CC_STAT_AREA]

        if area < 20:
            continue

        x = stats[label, cv2.CC_STAT_LEFT]
        y = stats[label, cv2.CC_STAT_TOP]
        bw = stats[label, cv2.CC_STAT_WIDTH]
        bh = stats[label, cv2.CC_STAT_HEIGHT]

        cx, cy = centroids[label]
        centroid = np.array([cx, cy])

        dist = np.linalg.norm(centroid - image_center)
        max_dist = np.linalg.norm(image_center)
        center_score = 1 - (dist / max_dist)

        touches_border = (
            x <= 1 or y <= 1 or
            x + bw >= w - 1 or
            y + bh >= h - 1
        )

        border_penalty = 0.5 if touches_border else 1.0

        score = area * center_score * border_penalty

        if score > best_score:
            best_score = score
            best_label = label

    if best_label is None:
        return mask.astype(np.uint8)

    final_mask = (labels == best_label).astype(np.uint8)

    return final_mask


def get_particle_mask(img_norm):
    """
    Generate particle mask using Otsu threshold.
    Since some particles may be darker and some brighter than background,
    both threshold directions are tested.
    """
    _, otsu = cv2.threshold(
        img_norm,
        0,
        255,
        cv2.THRESH_BINARY + cv2.THRESH_OTSU
    )

    candidate_bright = (img_norm > _).astype(np.uint8)
    candidate_dark = (img_norm < _).astype(np.uint8)

    candidate_bright = clean_mask(candidate_bright)
    candidate_dark = clean_mask(candidate_dark)

    candidate_bright = keep_best_component(candidate_bright)
    candidate_dark = keep_best_component(candidate_dark)

    def score_mask(mask):
        area_frac = mask.mean()

        if area_frac < 0.005 or area_frac > 0.90:
            return -1

        h, w = mask.shape
        ys, xs = np.where(mask > 0)

        if len(xs) == 0:
            return -1

        cx = xs.mean()
        cy = ys.mean()

        center_dist = np.sqrt((cx - w / 2) ** 2 + (cy - h / 2) ** 2)
        max_dist = np.sqrt((w / 2) ** 2 + (h / 2) ** 2)

        center_score = 1 - center_dist / max_dist

        return area_frac * center_score

    score_bright = score_mask(candidate_bright)
    score_dark = score_mask(candidate_dark)

    if score_bright >= score_dark:
        return candidate_bright
    else:
        return candidate_dark


def segment_internal_phases(img_norm, particle_mask, k=3):
    """
    Cluster pixel intensities inside the particle into k phase-like regions.
    This is visual/intensity clustering, not final chemical labeling.
    """
    phase_mask = np.zeros_like(img_norm, dtype=np.uint8)

    pixels = img_norm[particle_mask > 0].reshape(-1, 1)

    if len(pixels) < k:
        phase_mask[particle_mask > 0] = 1
        return phase_mask

    kmeans = KMeans(
        n_clusters=k,
        random_state=42,
        n_init=10
    )

    labels = kmeans.fit_predict(pixels)

    centers = kmeans.cluster_centers_.flatten()
    sorted_indices = np.argsort(centers)

    label_map = {}

    for new_label, old_label in enumerate(sorted_indices, start=1):
        label_map[old_label] = new_label

    remapped = np.array([label_map[x] for x in labels], dtype=np.uint8)

    phase_mask[particle_mask > 0] = remapped

    return phase_mask


def make_overlay(img_norm, phase_mask):
    """
    Create overlay for visual inspection.
    """
    img_rgb = cv2.cvtColor(img_norm, cv2.COLOR_GRAY2RGB)

    overlay = img_rgb.copy()

    colors = {
        1: np.array([0, 0, 255]),      # dark phase
        2: np.array([0, 255, 0]),      # medium phase
        3: np.array([255, 0, 0]),      # bright phase
    }

    alpha = 0.45

    for label, color in colors.items():
        region = phase_mask == label
        overlay[region] = (
            (1 - alpha) * overlay[region] + alpha * color
        ).astype(np.uint8)

    return overlay


def save_preview(img_norm, particle_mask, phase_mask, overlay, save_path):
    fig, axes = plt.subplots(1, 4, figsize=(16, 4))

    axes[0].imshow(img_norm, cmap="gray")
    axes[0].set_title("Original BSE")

    axes[1].imshow(particle_mask, cmap="gray")
    axes[1].set_title("Particle Mask")

    axes[2].imshow(phase_mask, cmap="tab10", vmin=0, vmax=3)
    axes[2].set_title("Phase Clusters")

    axes[3].imshow(overlay)
    axes[3].set_title("Overlay")

    for ax in axes:
        ax.axis("off")

    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()


# ----------------------------
# MAIN
# ----------------------------

all_images = []

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

print(f"Found {len(all_images)} images.")

preview_count = 0

for image_path in tqdm(all_images):
    img = read_grayscale(image_path)
    img_norm = normalize_image(img)

    particle_mask = get_particle_mask(img_norm)
    phase_mask = segment_internal_phases(img_norm, particle_mask, k=3)

    rel_path = os.path.relpath(image_path, ROOT_DIR)
    rel_path_no_ext = os.path.splitext(rel_path)[0]

    save_base = (
        rel_path_no_ext
        .replace(os.sep, "_")
        .replace(" ", "_")
        .replace(".", "_")
    )

    mask_save_path = os.path.join(MASK_DIR, save_base + "_phase_mask.png")
    preview_save_path = os.path.join(PREVIEW_DIR, save_base + "_preview.png")

    cv2.imwrite(mask_save_path, phase_mask)

    if preview_count < NUM_PREVIEWS:
        overlay = make_overlay(img_norm, phase_mask)
        save_preview(
            img_norm,
            particle_mask,
            phase_mask,
            overlay,
            preview_save_path
        )
        preview_count += 1

print("Done.")
print(f"Phase masks saved in: {MASK_DIR}")
print(f"Preview images saved in: {PREVIEW_DIR}")