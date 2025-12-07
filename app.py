import os
import io
import pickle
from typing import List
import sys

from flask import Flask, request, jsonify, render_template_string

import torch
from torch import nn
from torchvision import transforms, datasets
from PIL import Image

# Custom unpickler to force CPU device mapping
class CPUUnpickler(pickle.Unpickler):
    def find_class(self, module, name):
        if module == 'torch.storage' and name == '_load_from_bytes':
            return lambda x: torch.load(io.BytesIO(x), map_location=torch.device('cpu'), weights_only=False)
        return super().find_class(module, name)


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
IMG_HEIGHT = 128
IMG_WIDTH = 128
NUM_CLASSES = 38  # as in your training script

# Path to your trained model
MODEL_PATH = "model12.pkl"

# If available, we can infer class names from the dataset (optional)
DATASET_ROOT = "/root/.cache/kagglehub/datasets/kamal01/top-agriculture-crop-disease/versions/1/Crop Diseases/"

NORM_MEAN = [0.485, 0.456, 0.406]
NORM_STD = [0.229, 0.224, 0.225]

# Transforms for prediction (no augmentation)
test_transforms = transforms.Compose([
    transforms.Resize((IMG_HEIGHT, IMG_WIDTH)),
    transforms.ToTensor(),
    transforms.Normalize(NORM_MEAN, NORM_STD)
])

class CropClassifier(nn.Module):
    def __init__(self, num_classes: int):
        super(CropClassifier, self).__init__()
        self.conv_layer = nn.Sequential(
            # Layer 1: 3 -> 32 filters
            nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),  # 64x64

            # Layer 2: 32 -> 64 filters
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),  # 32x32

            # Layer 3: 64 -> 64 filters
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)   # 16x16
        )

        self.fc_layer = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(64 * 16 * 16, 256),
            nn.ReLU(),
            nn.Linear(256, num_classes)
        )

    def forward(self, x):
        x = self.conv_layer(x)
        x = x.reshape(x.size(0), -1)
        x = self.fc_layer(x)
        return x


def load_model(model_path: str) -> nn.Module:
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found at '{model_path}'")

    # Create architecture
    model = CropClassifier(NUM_CLASSES)

    # Load using custom unpickler that forces CPU device mapping
    with open(model_path, "rb") as f:
        unpickler = CPUUnpickler(f)
        loaded_model = unpickler.load()

    # If for some reason loaded_model is state_dict, we handle both cases:
    if isinstance(loaded_model, nn.Module):
        model = loaded_model
    else:
        model.load_state_dict(loaded_model)

    model.to(DEVICE)
    model.eval()
    return model


def infer_class_names(dataset_root: str, num_classes: int) -> List[str]:
    """
    Try to infer class names from the dataset directory using ImageFolder.
    If it fails, fall back to generic names.
    """
    try:
        if os.path.isdir(dataset_root):
            full_dataset = datasets.ImageFolder(root=dataset_root)
            class_names = [k.replace("_", " ").title() for k in full_dataset.classes]
            if len(class_names) == num_classes:
                print(f"[INFO] Inferred {len(class_names)} class names from dataset.")
                return class_names
            else:
                print(f"[WARN] Expected {num_classes} classes but found {len(class_names)} in dataset. Falling back.")
        else:
            print(f"[WARN] Dataset root '{dataset_root}' not found. Using generic class names.")
    except Exception as e:
        print(f"[WARN] Could not infer class names from dataset: {e}")

    # Fallback
    return [f"Class {i+1}" for i in range(num_classes)]


def predict_pil_image(img: Image.Image, model: nn.Module, class_names: List[str]) -> dict:
    img = img.convert("RGB")
    img_tensor = test_transforms(img)
    img_tensor = img_tensor.unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        outputs = model(img_tensor)
        logits = outputs
        probabilities = torch.softmax(logits, dim=1)
        confidence, predicted_index_tensor = torch.max(probabilities, 1)

        # also get top-k probabilities for debugging
        topk = min(5, probabilities.size(1))
        top_vals, top_idxs = torch.topk(probabilities, k=topk, dim=1)

    predicted_index = predicted_index_tensor.item()
    predicted_class = class_names[predicted_index]
    confidence_percent = confidence.item() * 100.0

    # Map confidence to a grade
    if confidence_percent >= 95.0:
        grade = "A"
        grade_description = "95-100% good crop"
    elif confidence_percent >= 85.0:
        grade = "B"
        grade_description = "85-95% good crop"
    elif confidence_percent >= 75.0:
        grade = "C"
        grade_description = "75-85% good crop"
    else:
        grade = "D"
        grade_description = "Below 75% - poor quality"
    # Debug log for prediction details including top-k probs and logits
    try:
        top_vals_list = [float(x) * 100.0 for x in top_vals[0].tolist()]
        top_idxs_list = [int(x) for x in top_idxs[0].tolist()]
        logits_list = [float(x) for x in logits[0].tolist()[:topk]]
        print(f"[PREDICT] class={predicted_class} index={predicted_index} conf={confidence_percent:.2f}% grade={grade}")
        print(f"[PREDICT] top_probs%={top_vals_list} top_idxs={top_idxs_list} sum_probs={float(probabilities.sum()):.6f}")
    except Exception:
        pass

    return {
        "predicted_class": predicted_class,
        "confidence": confidence_percent,
        "class_index": predicted_index,
        "grade": grade,
        "grade_description": grade_description
        ,"top_probs": top_vals_list if 'top_vals_list' in locals() else None
        ,"top_indices": top_idxs_list if 'top_idxs_list' in locals() else None
    }

app = Flask(__name__)

# Load model and class names once at startup
model = load_model(MODEL_PATH)
inferred_class_names = infer_class_names(DATASET_ROOT, NUM_CLASSES)


# Simple HTML upload page
HTML_PAGE = """
<!doctype html>
<html>
<head>
    <title>Crop Disease Classifier</title>
</head>
<body>
    <h1>Crop Disease Classifier</h1>
    <form action="/predict" method="post" enctype="multipart/form-data">
        <p>Select an image of a crop leaf:</p>
        <input type="file" name="file" accept="image/*" required>
        <button type="submit">Predict</button>
    </form>
    {% if result %}
        <h2>Result</h2>
        <p><strong>Predicted Condition:</strong> {{ result.predicted_class }}</p>
        <p><strong>Confidence:</strong> {{ "%.2f"|format(result.confidence) }}%</p>
        <p><strong>Grade:</strong> {{ result.grade }} - {{ result.grade_description }}</p>
    {% endif %}
</body>
</html>
"""


@app.route("/", methods=["GET"])
def index():
    return render_template_string(HTML_PAGE)


@app.route("/predict", methods=["POST"])
def predict_web():
    if "file" not in request.files:
        return render_template_string(HTML_PAGE, result=None)

    file = request.files["file"]
    if file.filename == "":
        return render_template_string(HTML_PAGE, result=None)

    try:
        img_bytes = file.read()
        img = Image.open(io.BytesIO(img_bytes))
        result = predict_pil_image(img, model, inferred_class_names)
        return render_template_string(HTML_PAGE, result=result)
    except Exception as e:
        return f"Error during prediction: {e}", 500


# JSON API endpoint
@app.route("/api/predict", methods=["POST"])
def api_predict():
    """
    Send a POST request with form-data:
        key: 'file', value: <image file>
    or raw file under the same field.
    """
    if "file" not in request.files:
        return jsonify({"error": "No file part in the request"}), 400

    file = request.files["file"]
    if file.filename == "":
        return jsonify({"error": "No file selected"}), 400

    try:
        img_bytes = file.read()
        img = Image.open(io.BytesIO(img_bytes))
        result = predict_pil_image(img, model, inferred_class_names)
        return jsonify({
            "predicted_class": result["predicted_class"],
            "confidence": result["confidence"],
            "class_index": result["class_index"],
            "grade": result.get("grade"),
            "grade_description": result.get("grade_description")
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    # You can set host="0.0.0.0" when deploying in a container or VM
    app.run(host="0.0.0.0", port=5000, debug=True)




# Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope Process; .\appt\Scripts\Activate.ps1