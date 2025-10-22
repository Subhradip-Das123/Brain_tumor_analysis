# ======================================================
# 🧠 Brain Tumor MRI Detection (PyTorch .pth version)
# Includes: GradCAM + 3D MRI Visualization
# Works with Google Drive or Local paths
# ======================================================

import os
import cv2
import numpy as np
import plotly.graph_objects as go
from PIL import Image
from scipy import ndimage
import torch
import torch.nn.functional as F
from torchvision import transforms
from torch import nn


# ======================================================
# 0️⃣ Setup constants
# ======================================================
IMAGE_SIZE = 128  # must match your training size
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


# ======================================================
# 1️⃣ Load model (Google Drive or local)
# ======================================================
def load_model_torch(model_path=None, model_class=None):
    """
    Loads a PyTorch model from a .pth file.
    model_class: a class defining your model architecture.
    """
    if model_path is None:
        try:
            from google.colab import drive
            drive.mount('/content/drive', force_remount=True)
            model_path = "/content/drive/MyDrive/brain_tumor_model.pth"
            print(f"✅ Using Google Drive model: {model_path}")
        except Exception:
            model_path = "model/brain_tumor_model.pth"
            print(f"✅ Using local model path: {model_path}")

    if not os.path.exists(model_path):
        raise FileNotFoundError(f"❌ Model not found at {model_path}")

    if model_class is None:
        raise ValueError("Please provide your model architecture class (model_class).")

    model = model_class()
    model.load_state_dict(torch.load(model_path, map_location=DEVICE))
    model.eval().to(DEVICE)
    print("✅ Model loaded successfully!")
    return model


# ======================================================
# 2️⃣ Preprocess and predict
# ======================================================
def preprocess_image(image: Image.Image):
    """
    Convert PIL image to tensor, normalize to [0,1],
    and resize to (IMAGE_SIZE, IMAGE_SIZE).
    """
    transform = transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.ToTensor(),
    ])
    return transform(image).unsqueeze(0).to(DEVICE)


@torch.no_grad()
def predict_tumor(model, x):
    """
    Run model prediction. Returns (pred_idx, confidence, probs)
    """
    logits = model(x)
    probs = F.softmax(logits, dim=1).cpu().numpy()[0]
    pred_idx = int(np.argmax(probs))
    confidence = float(np.max(probs))
    return pred_idx, confidence, probs


# ======================================================
# 3️⃣ Grad-CAM for PyTorch
# ======================================================
class GradCAM:
    def __init__(self, model, target_layer_name):
        self.model = model
        self.target_layer = self._get_target_layer(target_layer_name)
        self.gradients = None
        self.activations = None
        self._register_hooks()

    def _get_target_layer(self, name):
        """
        Gets a submodule by name.
        Example: 'features.29' for VGG or 'layer4.2.conv3' for ResNet.
        """
        layer = self.model
        for attr in name.split('.'):
            layer = getattr(layer, attr)
        return layer

    def _register_hooks(self):
        def forward_hook(module, input, output):
            self.activations = output.detach()

        def backward_hook(module, grad_input, grad_output):
            self.gradients = grad_output[0].detach()

        self.target_layer.register_forward_hook(forward_hook)
        self.target_layer.register_backward_hook(backward_hook)

    def generate(self, x, target_class=None):
        """
        Generates Grad-CAM heatmap for target class.
        """
        self.model.zero_grad()
        output = self.model(x)

        if target_class is None:
            target_class = output.argmax(dim=1).item()

        loss = output[0, target_class]
        loss.backward()

        # Compute Grad-CAM
        grads = self.gradients.mean(dim=[2, 3], keepdim=True)
        cam = (grads * self.activations).sum(dim=1, keepdim=True)
        cam = F.relu(cam)
        cam = F.interpolate(cam, size=(IMAGE_SIZE, IMAGE_SIZE), mode="bilinear", align_corners=False)
        cam = cam[0, 0].cpu().numpy()

        cam -= cam.min()
        cam /= (cam.max() + 1e-8)
        return cam


# ======================================================
# 4️⃣ Visualization functions
# ======================================================
def overlay_heatmap(img_bgr, heatmap, alpha=0.4):
    """Overlay heatmap on top of original image."""
    heatmap_uint8 = np.uint8(255 * heatmap)
    heatmap_color = cv2.applyColorMap(heatmap_uint8, cv2.COLORMAP_JET)
    overlay = cv2.addWeighted(img_bgr, 1 - alpha, heatmap_color, alpha, 0)
    return overlay


def make_pseudo3d(img_bgr, gradcam_heatmap=None, depth=24):
    """Create pseudo-3D MRI volume with tumor highlight."""
    img_gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    img_resized = cv2.resize(img_gray, (64, 64))
    img_norm = img_resized.astype(np.float32) / 255.0

    volume, tumor_mask = [], []
    center = depth // 2
    for i in range(depth):
        dist = abs(i - center) / max(center, 1)
        k = int(1 + dist * 3)
        if k % 2 == 0:
            k += 1
        blurred = cv2.GaussianBlur(img_norm, (k, k), 0)
        slice_img = blurred * (1.0 - 0.3 * dist)
        volume.append(slice_img)

        if gradcam_heatmap is not None:
            hm = cv2.resize(gradcam_heatmap, (64, 64))
            hm = hm / (hm.max() + 1e-8)
            tumor = hm > 0.5
            tumor_mask.append(tumor if abs(i - center) < depth // 3 else np.zeros_like(tumor))
        else:
            tumor_mask.append(np.zeros((64, 64), dtype=bool))

    return np.array(volume), np.array(tumor_mask)


def volume_to_fig(volume, tumor_mask=None, title="3D MRI Brain Visualization"):
    """Plotly 3D visualization."""
    depth, height, width = volume.shape
    step = 2
    vol_sub = volume[::step, ::step, ::step]
    D, H, W = vol_sub.shape

    x = np.linspace(0, 1, W)
    y = np.linspace(0, 1, H)
    z = np.linspace(0, 1, D)
    X, Y, Z = np.meshgrid(x, y, z, indexing='ij')

    fig = go.Figure()
    fig.add_trace(go.Volume(
        x=X.flatten(), y=Y.flatten(), z=Z.flatten(),
        value=vol_sub.flatten(),
        isomin=0.15, isomax=0.85,
        opacity=0.1, surface_count=10,
        colorscale='Gray', showscale=False
    ))

    if tumor_mask is not None and np.any(tumor_mask):
        t_sub = tumor_mask[::step, ::step, ::step]
        t_vals = t_sub.astype(float) * vol_sub
        mask = t_vals.flatten() > 0.1
        fig.add_trace(go.Volume(
            x=X.flatten()[mask], y=Y.flatten()[mask], z=Z.flatten()[mask],
            value=t_vals.flatten()[mask],
            isomin=0.1, isomax=1.0,
            opacity=0.4, surface_count=8,
            colorscale='Hot', showscale=True, name="Tumor Region"
        ))

    fig.update_layout(
        title=dict(text=title, x=0.5),
        width=800, height=700,
        scene=dict(aspectmode="cube"),
        margin=dict(l=0, r=0, b=0, t=40),
        paper_bgcolor='rgb(30,30,30)',
        font=dict(color='white')
    )
    return fig


# ======================================================
# 5️⃣ Example Usage
# ======================================================
if __name__ == "__main__":
    # 🧩 Define your model architecture (must match training)
    class BrainTumorNet(nn.Module):
        def __init__(self):
            super().__init__()
            self.features = nn.Sequential(
                nn.Conv2d(3, 16, 3, padding=1),
                nn.ReLU(),
                nn.MaxPool2d(2),
                nn.Conv2d(16, 32, 3, padding=1),
                nn.ReLU(),
                nn.MaxPool2d(2)
            )
            self.classifier = nn.Sequential(
                nn.Flatten(),
                nn.Linear(32 * 32 * 32, 64),
                nn.ReLU(),
                nn.Linear(64, 2)  # binary classification
            )

        def forward(self, x):
            x = self.features(x)
            x = self.classifier(x)
            return x

    # 1️⃣ Load model
    model = load_model_torch(model_class=BrainTumorNet)

    # 2️⃣ Load image
    image_path = "/content/drive/MyDrive/MRI/Training/pituitary/Tr-pi_0010.jpg"  # or local "test_mri.jpg"
    image = Image.open(image_path)

    # 3️⃣ Preprocess
    x = preprocess_image(image)

    # 4️⃣ Predict
    pred_idx, confidence, probs = predict_tumor(model, x)
    print(f"\n🩺 Prediction: Class {pred_idx}, Confidence {confidence:.2f}")

    # 5️⃣ Grad-CAM
    cam_gen = GradCAM(model, target_layer_name="features.3")  # pick your conv layer index
    cam = cam_gen.generate(x)

    # 6️⃣ Overlay Heatmap
    img_bgr = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
    overlay = overlay_heatmap(img_bgr, cam)
    cv2.imwrite("gradcam_overlay.jpg", overlay)
    print("✅ Grad-CAM overlay saved as gradcam_overlay.jpg")

    # 7️⃣ 3D Visualization
    volume, tumor_mask = make_pseudo3d(img_bgr, cam)
    fig = volume_to_fig(volume, tumor_mask)
    fig.show()

