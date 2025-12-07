import os
from io import BytesIO

from flask import Flask, request, render_template_string
from PIL import Image

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms

def ConvBlock(in_channels, out_channels, pool=False):
    layers = [
        nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(inplace=True),
    ]
    if pool:
        layers.append(nn.MaxPool2d(4))
    return nn.Sequential(*layers)


class ImageClassificationBase(nn.Module):
    def training_step(self, batch):
        images, labels = batch
        out = self(images)
        loss = F.cross_entropy(out, labels)
        return loss

    def validation_step(self, batch):
        images, labels = batch
        out = self(images)
        loss = F.cross_entropy(out, labels)
        _, preds = torch.max(out, dim=1)
        acc = torch.tensor(torch.sum(preds == labels).item() / len(preds))
        return {"val_loss": loss.detach(), "val_acc": acc}

    def validation_epoch_end(self, outputs):
        batch_losses = [x["val_loss"] for x in outputs]
        epoch_loss = torch.stack(batch_losses).mean()
        batch_accs = [x["val_acc"] for x in outputs]
        epoch_acc = torch.stack(batch_accs).mean()
        return {"val_loss": epoch_loss.item(), "val_acc": epoch_acc.item()}

    def epoch_end(self, epoch, result):
        print(
            "Epoch [{}], train_loss: {:.4f}, val_loss: {:.4f}, val_acc: {:.4f}".format(
                epoch, result["train_loss"], result["val_loss"], result["val_acc"]
            )
        )


class CNN_NeuralNet(ImageClassificationBase):
    def __init__(self, in_channels, num_diseases):
        super().__init__()

        self.conv1 = ConvBlock(in_channels, 64)
        self.conv2 = ConvBlock(64, 128, pool=True)
        self.res1 = nn.Sequential(ConvBlock(128, 128), ConvBlock(128, 128))

        self.conv3 = ConvBlock(128, 256, pool=True)
        self.conv4 = ConvBlock(256, 512, pool=True)

        self.res2 = nn.Sequential(ConvBlock(512, 512), ConvBlock(512, 512))
        self.classifier = nn.Sequential(
            nn.MaxPool2d(4),
            nn.Flatten(),
            nn.Linear(512, num_diseases),
        )

    def forward(self, x):
        out = self.conv1(x)
        out = self.conv2(out)
        out = self.res1(out) + out
        out = self.conv3(out)
        out = self.conv4(out)
        out = self.res2(out) + out
        out = self.classifier(out)
        return out

# ⚠️ IMPORTANT: Put your disease / plant classes here in the SAME ORDER
# as train.classes from your training script.
# Example (replace with your actual list):
CLASSES = [
    # "Apple___Apple_scab",
    # "Apple___Black_rot",
    # "Apple___Cedar_apple_rust",
    # ...
]

NUM_CLASSES = len(CLASSES)

if NUM_CLASSES == 0:
    print(
        "[WARNING] CLASSES list is empty. "
        "Fill CLASSES with your class names in the right order!"
    )

# Device (GPU if available, else CPU)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {DEVICE}")

# Image transforms (same as used during training/testing)
transform_for_prediction = transforms.Compose(
    [
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
    ]
)

# Path to your saved model weights
MODEL_PATH = "plant_disease_model.pth"  # change if your filename is different

# Create model and load weights
model = CNN_NeuralNet(in_channels=3, num_diseases=NUM_CLASSES).to(DEVICE)

if os.path.exists(MODEL_PATH):
    state_dict = torch.load(MODEL_PATH, map_location=DEVICE)
    model.load_state_dict(state_dict)
    model.eval()
    print(f"Loaded model weights from {MODEL_PATH}")
else:
    print(
        f"[WARNING] Model file '{MODEL_PATH}' not found. "
        "Place your trained weights in this path."
    )


# =========================================================
# 3. Prediction helper
# =========================================================

def predict_image_pil(image: Image.Image):
    """
    Takes a PIL image, applies transforms, runs through the model,
    and returns predicted class label and probability.
    """
    # Apply transforms
    img_t = transform_for_prediction(image).unsqueeze(0)  # shape (1, C, H, W)
    img_t = img_t.to(DEVICE)

    with torch.no_grad():
        outputs = model(img_t)
        probs = torch.softmax(outputs, dim=1)
        conf, preds = torch.max(probs, dim=1)

    class_idx = preds.item()
    class_name = CLASSES[class_idx] if 0 <= class_idx < len(CLASSES) else str(class_idx)
    confidence = conf.item()

    return class_name, confidence


# =========================================================
# 4. Flask app
# =========================================================

app = Flask(__name__)

# Simple HTML template (no separate templates folder needed)
HTML_TEMPLATE = """
<!doctype html>
<html lang="en">
  <head>
    <title>Plant Disease Detection</title>
    <meta charset="utf-8">
    <style>
      body { font-family: Arial, sans-serif; margin: 40px; }
      h1 { color: #2E7D32; }
      .container { max-width: 600px; margin: auto; }
      .result { margin-top: 20px; padding: 15px; border-radius: 8px; background: #f3f3f3; }
      .error { color: red; }
      .preview-img { max-width: 100%; height: auto; margin-top: 10px; border-radius: 4px; }
      .footer { margin-top: 40px; font-size: 12px; color: #888; }
      button { background: #2E7D32; color: white; border: none; padding: 10px 15px;
               border-radius: 4px; cursor: pointer; }
      button:hover { background: #1B5E20; }
      input[type=file] { margin-top: 10px; }
    </style>
  </head>
  <body>
    <div class="container">
      <h1>🌱 Plant Disease Detection</h1>
      <p>Upload a leaf image and the model will predict the disease class.</p>
      <form method="POST" enctype="multipart/form-data">
        <input type="file" name="image" accept="image/*" required>
        <br><br>
        <button type="submit">Predict</button>
      </form>

      {% if error %}
        <div class="result">
          <p class="error">{{ error }}</p>
        </div>
      {% endif %}

      {% if predicted_class %}
        <div class="result">
          <h3>Prediction:</h3>
          <p><b>Class:</b> {{ predicted_class }}</p>
          <p><b>Confidence:</b> {{ confidence | round(4) }}</p>
          {% if image_data %}
            <h4>Uploaded Image:</h4>
            <img src="data:image/png;base64,{{ image_data }}" class="preview-img">
          {% endif %}
        </div>
      {% endif %}

      <div class="footer">
        Running on Flask (localhost). Make sure model weights and class names are correctly set in <code>app.py</code>.
      </div>
    </div>
  </body>
</html>
"""

import base64

@app.route("/", methods=["GET", "POST"])
def index():
    error = None
    predicted_class = None
    confidence = None
    image_data_b64 = None

    if request.method == "POST":
        if "image" not in request.files:
            error = "No file part in the request."
        else:
            file = request.files["image"]
            if file.filename == "":
                error = "No file selected."
            else:
                try:
                    # Read image from the uploaded file
                    image_bytes = file.read()
                    image = Image.open(BytesIO(image_bytes)).convert("RGB")

                    # Predict
                    predicted_class, confidence = predict_image_pil(image)

                    # Convert image to base64 to show preview on page
                    buffer = BytesIO()
                    image.save(buffer, format="PNG")
                    image_data_b64 = base64.b64encode(buffer.getvalue()).decode("utf-8")

                except Exception as e:
                    error = f"Error processing the image: {e}"

    return render_template_string(
        HTML_TEMPLATE,
        error=error,
        predicted_class=predicted_class,
        confidence=confidence,
        image_data=image_data_b64,
    )


if __name__ == "__main__":
    # Run on localhost:5000
    app.run(host="0.0.0.0", port=5000, debug=True)