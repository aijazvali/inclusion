import os
import cv2
import joblib
import numpy as np
import pandas as pd
import gradio as gr

import torch
import torch.nn as nn

from sklearn.cluster import KMeans


# ----------------------------
# CONFIG
# ----------------------------

MODEL_PATH = "frontend_model/combined_3class_frontend_model.pt"
PREPROCESS_PATH = "frontend_model/combined_3class_preprocess.joblib"

IMG_SIZE = 128

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ----------------------------
# MODEL
# ----------------------------

class CombinedCNNTabular(nn.Module):
    def __init__(self, num_tab_features, num_classes):
        super().__init__()

        self.image_encoder = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),

            nn.Conv2d(32, 64, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),

            nn.Conv2d(64, 128, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),

            nn.Conv2d(128, 256, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),

            nn.Conv2d(256, 384, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(384),
            nn.ReLU(inplace=True),

            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
        )

        self.tabular_encoder = nn.Sequential(
            nn.Linear(num_tab_features, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.20),

            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(inplace=True),
        )

        self.classifier = nn.Sequential(
            nn.Linear(384 + 64, 192),
            nn.ReLU(inplace=True),
            nn.Dropout(0.35),
            nn.Linear(192, num_classes),
        )

    def forward(self, img, tab):
        img_feat = self.image_encoder(img)
        tab_feat = self.tabular_encoder(tab)
        x = torch.cat([img_feat, tab_feat], dim=1)
        return self.classifier(x)


# ----------------------------
# IMAGE PROCESSING
# ----------------------------

def normalize_image_uint8(img):
    img = img.astype(np.float32)

    low, high = np.percentile(img, [1, 99])

    if high <= low:
        img = img / 255.0
        img = np.clip(img, 0, 1)
        return (img * 255).astype(np.uint8)

    img = (img - low) / (high - low)
    img = np.clip(img, 0, 1)

    return (img * 255).astype(np.uint8)


def normalize_image_float(img):
    img = img.astype(np.float32)

    low, high = np.percentile(img, [1, 99])

    if high <= low:
        return np.clip(img / 255.0, 0, 1)

    img = (img - low) / (high - low)
    img = np.clip(img, 0, 1)

    return img


def clean_mask(mask):
    mask = mask.astype(np.uint8)
    kernel = np.ones((3, 3), np.uint8)

    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)

    return mask


def keep_best_component(mask):
    h, w = mask.shape

    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(
        mask.astype(np.uint8),
        connectivity=8,
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
        center_score = 1 - dist / max_dist

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

    return (labels == best_label).astype(np.uint8)


def get_particle_mask(img_norm):
    threshold_value, _ = cv2.threshold(
        img_norm,
        0,
        255,
        cv2.THRESH_BINARY + cv2.THRESH_OTSU,
    )

    candidate_bright = (img_norm > threshold_value).astype(np.uint8)
    candidate_dark = (img_norm < threshold_value).astype(np.uint8)

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

    return candidate_dark


def segment_internal_phases(img_norm, particle_mask, k=3):
    phase_mask = np.zeros_like(img_norm, dtype=np.uint8)

    pixels = img_norm[particle_mask > 0].reshape(-1, 1)

    if len(pixels) < k:
        phase_mask[particle_mask > 0] = 1
        return phase_mask

    kmeans = KMeans(
        n_clusters=k,
        random_state=42,
        n_init=10,
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
    img_rgb = cv2.cvtColor(img_norm, cv2.COLOR_GRAY2RGB)
    overlay = img_rgb.copy()

    colors = {
        1: np.array([0, 0, 255]),
        2: np.array([0, 255, 0]),
        3: np.array([255, 0, 0]),
    }

    alpha = 0.45

    for label, color in colors.items():
        region = phase_mask == label
        overlay[region] = ((1 - alpha) * overlay[region] + alpha * color).astype(np.uint8)

    return overlay


def phase_mask_to_color(phase_mask):
    palette = np.array([
        [0, 0, 0],
        [0, 0, 255],
        [0, 255, 0],
        [255, 0, 0],
    ], dtype=np.uint8)

    return palette[phase_mask]


# ----------------------------
# FEATURE EXTRACTION
# ----------------------------

def count_regions(binary_mask):
    binary_mask = binary_mask.astype(np.uint8)

    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(
        binary_mask,
        connectivity=8,
    )

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


def extract_features(img_norm_uint8, phase_mask):
    particle_mask = (phase_mask > 0).astype(np.uint8)

    image_area = phase_mask.shape[0] * phase_mask.shape[1]
    particle_area = int(particle_mask.sum())

    row = {
        "image_h": phase_mask.shape[0],
        "image_w": phase_mask.shape[1],
        "image_area": image_area,
        "particle_area": particle_area,
        "particle_area_fraction": particle_area / image_area if image_area > 0 else 0,
    }

    row.update(bbox_features(particle_mask))

    if particle_area > 0:
        particle_pixels = img_norm_uint8[particle_mask > 0]
        row["particle_mean_intensity"] = float(np.mean(particle_pixels))
        row["particle_std_intensity"] = float(np.std(particle_pixels))
        row["particle_min_intensity"] = int(np.min(particle_pixels))
        row["particle_max_intensity"] = int(np.max(particle_pixels))
    else:
        row["particle_mean_intensity"] = 0
        row["particle_std_intensity"] = 0
        row["particle_min_intensity"] = 0
        row["particle_max_intensity"] = 0

    total_regions = 0

    for phase in [1, 2, 3]:
        phase_binary = (phase_mask == phase).astype(np.uint8)
        phase_area = int(phase_binary.sum())

        row[f"phase_{phase}_area"] = phase_area
        row[f"phase_{phase}_area_fraction_image"] = phase_area / image_area if image_area > 0 else 0
        row[f"phase_{phase}_area_fraction_particle"] = (
            phase_area / particle_area if particle_area > 0 else 0
        )

        if phase_area > 0:
            pixels = img_norm_uint8[phase_binary > 0]
            row[f"phase_{phase}_mean_intensity"] = float(np.mean(pixels))
            row[f"phase_{phase}_std_intensity"] = float(np.std(pixels))
            row[f"phase_{phase}_min_intensity"] = int(np.min(pixels))
            row[f"phase_{phase}_max_intensity"] = int(np.max(pixels))
        else:
            row[f"phase_{phase}_mean_intensity"] = 0
            row[f"phase_{phase}_std_intensity"] = 0
            row[f"phase_{phase}_min_intensity"] = 0
            row[f"phase_{phase}_max_intensity"] = 0

        num_regions, largest_region = count_regions(phase_binary)

        row[f"phase_{phase}_num_regions"] = num_regions
        row[f"phase_{phase}_largest_region_area"] = largest_region

        total_regions += num_regions

    row["total_phase_regions"] = total_regions

    return row


# ----------------------------
# LOAD MODEL
# ----------------------------

checkpoint = torch.load(MODEL_PATH, map_location=DEVICE)
preprocess = joblib.load(PREPROCESS_PATH)

classes = checkpoint["classes"]
feature_cols = preprocess["feature_cols"]
imputer = preprocess["imputer"]
scaler = preprocess["scaler"]

model = CombinedCNNTabular(
    num_tab_features=len(feature_cols),
    num_classes=len(classes),
)

model.load_state_dict(checkpoint["model_state_dict"])
model.to(DEVICE)
model.eval()


# ----------------------------
# PREDICTION FUNCTION
# ----------------------------

def predict_inclusion_type(input_image):
    if input_image is None:
        return None, None, None, None, "Please upload an image."

    img = input_image

    if img.ndim == 3:
        img_gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    else:
        img_gray = img.copy()

    img_gray = cv2.resize(img_gray, (IMG_SIZE, IMG_SIZE), interpolation=cv2.INTER_AREA)

    img_uint8 = normalize_image_uint8(img_gray)
    img_float = normalize_image_float(img_gray)

    particle_mask = get_particle_mask(img_uint8)
    phase_mask = segment_internal_phases(img_uint8, particle_mask, k=3)

    overlay = make_overlay(img_uint8, phase_mask)
    color_mask = phase_mask_to_color(phase_mask)

    features = extract_features(img_uint8, phase_mask)
    feature_df = pd.DataFrame([features])

    for col in feature_cols:
        if col not in feature_df.columns:
            feature_df[col] = 0

    X_tab = feature_df[feature_cols]
    X_tab = imputer.transform(X_tab)
    X_tab = scaler.transform(X_tab)

    img_tensor = torch.tensor(
        img_float.tolist(),
        dtype=torch.float32,
        device=DEVICE
    ).unsqueeze(0).unsqueeze(0)

    tab_tensor = torch.tensor(
        X_tab.astype(np.float32).tolist(),
        dtype=torch.float32,
        device=DEVICE
    )

    with torch.no_grad():
        logits = model(img_tensor, tab_tensor)
        probs = torch.softmax(logits, dim=1).detach().cpu().tolist()[0]

    pred_idx = probs.index(max(probs))
    pred_class = classes[pred_idx]
    confidence = float(probs[pred_idx])

    prob_df = pd.DataFrame({
        "class": classes,
        "probability": probs,
    }).sort_values("probability", ascending=False)

    particle_area = features["particle_area"]
    phase_1_frac = features["phase_1_area_fraction_particle"]
    phase_2_frac = features["phase_2_area_fraction_particle"]
    phase_3_frac = features["phase_3_area_fraction_particle"]

    summary = f"""
Prediction: {pred_class}

Confidence: {confidence:.4f}

Particle area: {particle_area} pixels

Phase fractions inside particle:
- Phase 1 dark: {phase_1_frac:.3f}
- Phase 2 medium: {phase_2_frac:.3f}
- Phase 3 bright: {phase_3_frac:.3f}

Note: This model predicts only the 3 broad inclusion types trained so far:
oxide, oxy-sulfide, sulfide.
"""

    return img_uint8, color_mask, overlay, prob_df, summary


# ----------------------------
# GRADIO UI
# ----------------------------

with gr.Blocks(title="SEM Inclusion Type Classifier") as demo:
    gr.Markdown(
        """
        # SEM-BSE Inclusion Type Classifier

        Upload a particle crop image.  
        The app generates a phase segmentation mask, extracts phase features, and predicts the broad inclusion type.

        Current supported classes:

        **oxide / oxy-sulfide / sulfide**
        """
    )

    with gr.Row():
        with gr.Column():
            input_image = gr.Image(
                label="Upload SEM-BSE particle crop",
                type="numpy",
            )

            run_btn = gr.Button("Analyze Inclusion", variant="primary")

        with gr.Column():
            summary_box = gr.Textbox(
                label="Prediction Summary",
                lines=12,
            )

            prob_table = gr.Dataframe(
                label="Class Probabilities",
                headers=["class", "probability"],
            )

    with gr.Row():
        original_out = gr.Image(label="Normalized Input")
        mask_out = gr.Image(label="Phase Mask")
        overlay_out = gr.Image(label="Overlay")

    run_btn.click(
        fn=predict_inclusion_type,
        inputs=input_image,
        outputs=[
            original_out,
            mask_out,
            overlay_out,
            prob_table,
            summary_box,
        ],
    )

demo.launch(share="true")