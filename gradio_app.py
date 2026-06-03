from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from PIL import Image

import gradio as gr
from torchvision import transforms


# ==================================================
# 1. Configuration
# ==================================================

MODEL_PATHS = {
    "Fold 1 model": "best_inclusion_cnn_fold1.pt",
    "Fold 2 model": "best_inclusion_cnn_fold2.pt",
    "Fold 3 model": "best_inclusion_cnn_fold3.pt",
    "Fold 4 model": "best_inclusion_cnn_fold4.pt",
}

ENSEMBLE_OPTION = "Ensemble: average all 4 models"


# ==================================================
# 2. Device
# ==================================================

if torch.cuda.is_available():
    device = torch.device("cuda")
elif torch.backends.mps.is_available():
    device = torch.device("mps")
else:
    device = torch.device("cpu")

print("Using device:", device)


# ==================================================
# 3. Model definition
# ==================================================

class InclusionCNN(nn.Module):
    def __init__(self, num_classes=2):
        super().__init__()

        self.features = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),

            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),

            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),

            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),

            nn.Conv2d(256, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((1, 1)),
        )

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(0.3),
            nn.Linear(512, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(128, num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x


# ==================================================
# 4. Load checkpoints
# ==================================================

def safe_torch_load(path, map_location):
    try:
        return torch.load(path, map_location=map_location, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=map_location)


def load_model(checkpoint_path):
    checkpoint_path = Path(checkpoint_path)

    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Model checkpoint not found: {checkpoint_path}")

    checkpoint = safe_torch_load(checkpoint_path, map_location=device)

    model = InclusionCNN(num_classes=2).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    mean = checkpoint["mean"]
    std = checkpoint["std"]

    metadata = {
        "path": str(checkpoint_path),
        "mean": mean,
        "std": std,
        "fold_column": checkpoint.get("fold_column", "unknown"),
        "epoch": checkpoint.get("epoch", "unknown"),
        "best_val_balanced_accuracy": checkpoint.get(
            "best_val_balanced_accuracy",
            "unknown",
        ),
    }

    return model, metadata


loaded_models = {}

for name, path in MODEL_PATHS.items():
    path_obj = Path(path)

    if path_obj.exists():
        model, metadata = load_model(path_obj)
        loaded_models[name] = {
            "model": model,
            "metadata": metadata,
        }
        print(f"Loaded {name}: {path}")
    else:
        print(f"Missing {name}: {path}")


if len(loaded_models) == 0:
    raise RuntimeError(
        "No model checkpoints found. Make sure best_inclusion_cnn_fold1.pt etc. "
        "are in the same folder as gradio_app.py"
    )


# ==================================================
# 5. Prediction helpers
# ==================================================

def preprocess_image(pil_image, mean, std):
    if pil_image is None:
        raise ValueError("No image uploaded.")

    # SEM-BSE images are grayscale. Force uploaded image to grayscale.
    image = pil_image.convert("L")

    # Model was trained on 128x128 crops.
    image = image.resize((128, 128))

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[mean], std=[std]),
    ])

    tensor = transform(image)
    tensor = tensor.unsqueeze(0)  # [1, 1, 128, 128]

    return tensor.to(device)


def predict_with_one_model(pil_image, model_name):
    item = loaded_models[model_name]
    model = item["model"]
    metadata = item["metadata"]

    x = preprocess_image(
        pil_image,
        mean=metadata["mean"],
        std=metadata["std"],
    )

    with torch.no_grad():
        logits = model(x)
        probs = torch.softmax(logits, dim=1)

    prob_non_inclusion = float(probs[0, 0].detach().cpu().item())
    prob_inclusion = float(probs[0, 1].detach().cpu().item())

    return prob_non_inclusion, prob_inclusion


def predict(pil_image, model_choice, threshold):
    if pil_image is None:
        return (
            "Please upload an image.",
            None,
            None,
            [["error", "No image uploaded"]],
        )

    if model_choice == ENSEMBLE_OPTION:
        probs_non = []
        probs_inc = []

        for name in loaded_models.keys():
            p0, p1 = predict_with_one_model(pil_image, name)
            probs_non.append(p0)
            probs_inc.append(p1)

        prob_non_inclusion = float(np.mean(probs_non))
        prob_inclusion = float(np.mean(probs_inc))

        model_used = "Average of all loaded fold models"

    else:
        prob_non_inclusion, prob_inclusion = predict_with_one_model(
            pil_image,
            model_choice,
        )
        model_used = model_choice

    pred_label = "inclusion" if prob_inclusion >= threshold else "non-inclusion"
    confidence = prob_inclusion if pred_label == "inclusion" else prob_non_inclusion

    if pred_label == "inclusion":
        verdict = "Prediction: INCLUSION"
    else:
        verdict = "Prediction: NON-INCLUSION"

    result_text = (
        f"{verdict}\n\n"
        f"Probability of inclusion: {prob_inclusion:.4f}\n"
        f"Probability of non-inclusion: {prob_non_inclusion:.4f}\n"
        f"Threshold: {threshold:.2f}\n"
        f"Confidence: {confidence:.4f}\n"
        f"Model used: {model_used}"
    )

    label_scores = {
        "inclusion": prob_inclusion,
        "non-inclusion": prob_non_inclusion,
    }

    details = [
        ["model_used", model_used],
        ["predicted_label", pred_label],
        ["threshold", f"{threshold:.2f}"],
        ["prob_inclusion", f"{prob_inclusion:.4f}"],
        ["prob_non_inclusion", f"{prob_non_inclusion:.4f}"],
        ["confidence", f"{confidence:.4f}"],
    ]

    return result_text, label_scores, pil_image.resize((256, 256)), details


# ==================================================
# 6. Gradio UI
# ==================================================

model_choices = list(loaded_models.keys())

if len(loaded_models) > 1:
    model_choices = [ENSEMBLE_OPTION] + model_choices

with gr.Blocks() as demo:
    gr.Markdown(
        """
        # Steel Inclusion Classifier

        Upload a **128×128 SEM-BSE particle crop** and the model will classify it as:

        - `inclusion`
        - `non-inclusion`

        This is the binary classifier frontend. It does not yet perform detection or segmentation on full SEM images.
        """
    )

    with gr.Row():
        with gr.Column():
            image_input = gr.Image(
                label="Upload SEM-BSE particle crop",
                type="pil",
            )

            model_dropdown = gr.Dropdown(
                choices=model_choices,
                value=model_choices[0],
                label="Model",
            )

            threshold_slider = gr.Slider(
                minimum=0.05,
                maximum=0.95,
                value=0.50,
                step=0.01,
                label="Inclusion threshold",
            )

            predict_button = gr.Button("Predict")

        with gr.Column():
            result_output = gr.Textbox(
                label="Prediction result",
                lines=8,
            )

            label_output = gr.Label(
                label="Class probabilities",
                num_top_classes=2,
            )

            preview_output = gr.Image(
                label="Model input preview",
                type="pil",
            )

            details_output = gr.Dataframe(
                headers=["Field", "Value"],
                label="Details",
            )

    predict_button.click(
        fn=predict,
        inputs=[
            image_input,
            model_dropdown,
            threshold_slider,
        ],
        outputs=[
            result_output,
            label_output,
            preview_output,
            details_output,
        ],
    )

    gr.Markdown(
        """
        ## Notes

        - Use threshold `0.50` for balanced reporting.
        - Lower threshold catches more inclusions but increases false positives.
        - The image is converted to grayscale and resized to `128×128` before prediction.
        """
    )


if __name__ == "__main__":
    demo.launch()