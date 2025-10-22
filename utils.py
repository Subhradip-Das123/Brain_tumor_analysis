import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
import os
import numpy as np
import cv2
from PIL import Image
import plotly.graph_objects as go
from scipy import ndimage


# ======================================================
# 1️⃣ Load Model (Handles both state_dict & full model)
# ======================================================
def load_model(model_path="model/brain_tumor_model.pth"):
    """
    Load a trained VGG16 brain tumor model.
    Works whether saved via torch.save(model) or torch.save(model.state_dict()).
    """
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"❌ Model not found at: {model_path}")

    try:
        # Try loading full model first
        model = torch.load(model_path, map_location=torch.device('cpu'))
        model.eval()
        print("✅ Loaded full model successfully.")
        return model

    except Exception:
        print("⚠️ Detected state_dict format — rebuilding VGG16 architecture...")
        model = models.vgg16(pretrained=False)
        model.classifier[6] = nn.Linear(4096, 3)  # 3 output classes: Glioma, Meningioma, Pituitary

        state_dict = torch.load(model_path, map_location=torch.device('cpu'))

        # Fix multi-GPU 'module.' prefix if present
        if any(k.startswith("module.") for k in state_dict.keys()):
            from collections import OrderedDict
            new_state_dict = OrderedDict()
            for k, v in state_dict.items():
                new_state_dict[k.replace("module.", "")] = v
            state_dict = new_state_dict

        model.load_state_dict(state_dict)
        model.eval()
        print("✅ Loaded model weights successfully.")
        return model


# ======================================================
# 2️⃣ Image Preprocessing & Prediction
# ======================================================
def preprocess_image(image: Image.Image):
    """
    Apply the same preprocessing used during model training.
    Converts PIL image to tensor (1, 3, 224, 224).
    """
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225])
    ])
    return transform(image).unsqueeze(0)


def predict_tumor(model, image_tensor):
    """
    Run inference and return (predicted_label_index, confidence_score).
    """
    with torch.no_grad():
        outputs = model(image_tensor)
        probs = torch.nn.functional.softmax(outputs, dim=1)[0]
        predicted = torch.argmax(probs).item()
        return predicted, probs[predicted].item()


# ======================================================
# 3️⃣ GradCAM Heatmap Overlay
# ======================================================
def overlay_heatmap(img_bgr, heatmap, alpha=0.4):
    """
    Overlay a heatmap on the original MRI image.
    """
    heatmap_resized = cv2.resize(heatmap, (img_bgr.shape[1], img_bgr.shape[0]))
    heatmap_normalized = np.uint8(255 * heatmap_resized / (heatmap_resized.max() + 1e-8))
    heatmap_colored = cv2.applyColorMap(heatmap_normalized, cv2.COLORMAP_JET)
    overlay = cv2.addWeighted(img_bgr, 1 - alpha, heatmap_colored, alpha, 0)
    return overlay


# ======================================================
# 4️⃣ Generate Pseudo-3D MRI Volume
# ======================================================
def make_pseudo3d(img_bgr, gradcam_heatmap=None, depth=24):
    """
    Create a pseudo-3D MRI volume with optional GradCAM-based tumor highlighting.
    """
    img_gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    img_resized = cv2.resize(img_gray, (64, 64))
    img_normalized = img_resized.astype(np.float32) / 255.0

    volume, tumor_mask = [], []
    center_slice = depth // 2

    for i in range(depth):
        distance = abs(i - center_slice) / center_slice
        blur_amount = int(1 + distance * 3)
        if blur_amount % 2 == 0:
            blur_amount += 1

        blurred = cv2.GaussianBlur(img_normalized, (blur_amount, blur_amount), 0)
        intensity_factor = 1.0 - (distance * 0.3)
        slice_img = blurred * intensity_factor
        volume.append(slice_img)

        if gradcam_heatmap is not None:
            heatmap_resized = cv2.resize(gradcam_heatmap, (64, 64))
            heatmap_norm = heatmap_resized / (heatmap_resized.max() + 1e-8)
            tumor_slice = heatmap_norm > 0.5

            if abs(i - center_slice) < depth // 3:
                tumor_mask.append(tumor_slice)
            else:
                tumor_mask.append(np.zeros_like(tumor_slice, dtype=bool))
        else:
            tumor_mask.append(np.zeros((64, 64), dtype=bool))

    return np.array(volume), np.array(tumor_mask)


# ======================================================
# 5️⃣ 3D MRI Visualization (Plotly)
# ======================================================
def volume_to_html(volume, tumor_mask=None, title="3D MRI Visualization"):
    """
    Convert a 3D numpy MRI volume into interactive HTML (Plotly).
    Optimized for web rendering performance.
    """
    depth, height, width = volume.shape
    step = 2
    volume_sub = volume[::step, ::step, ::step]
    depth_sub, height_sub, width_sub = volume_sub.shape

    x = np.linspace(0, 1, width_sub)
    y = np.linspace(0, 1, height_sub)
    z = np.linspace(0, 1, depth_sub)
    X, Y, Z = np.meshgrid(x, y, z, indexing='ij')

    x_flat, y_flat, z_flat = X.flatten(), Y.flatten(), Z.flatten()
    values = volume_sub.flatten()

    fig = go.Figure()

    # Main brain tissue
    fig.add_trace(go.Volume(
        x=x_flat, y=y_flat, z=z_flat, value=values,
        isomin=0.15, isomax=0.85, opacity=0.1,
        surface_count=10, colorscale='Gray', showscale=False,
        caps=dict(x_show=False, y_show=False, z_show=False)
    ))

    # Tumor region overlay
    if tumor_mask is not None and np.any(tumor_mask):
        tumor_sub = tumor_mask[::step, ::step, ::step]
        tumor_values = tumor_sub.astype(float) * volume_sub
        tumor_flat = tumor_values.flatten()
        tumor_indices = tumor_flat > 0.1

        if np.any(tumor_indices):
            fig.add_trace(go.Volume(
                x=x_flat[tumor_indices],
                y=y_flat[tumor_indices],
                z=z_flat[tumor_indices],
                value=tumor_flat[tumor_indices],
                isomin=0.1, isomax=1.0, opacity=0.4,
                surface_count=8, colorscale='Hot',
                name='Tumor Region', showscale=True,
                colorbar=dict(title="Intensity", x=1.05, len=0.7)
            ))

    fig.update_layout(
        title=dict(text=title, x=0.5, font=dict(size=16)),
        width=600, height=600,
        margin=dict(l=0, r=0, b=0, t=40),
        paper_bgcolor='rgb(30, 30, 30)',
        font=dict(color='white')
    )

    return fig.to_html(
        full_html=False,
        include_plotlyjs='cdn',
        config={'displayModeBar': True, 'displaylogo': False}
    )


# ======================================================
# 6️⃣ Load NIfTI 3D Volume (.nii, .nii.gz)
# ======================================================
def load_nifti_volume(nifti_path):
    """Load a 3D NIfTI MRI volume."""
    try:
        import nibabel as nib
        nii_img = nib.load(nifti_path)
        volume = nii_img.get_fdata()
        volume = (volume - volume.min()) / (volume.max() - volume.min())

        if volume.shape[0] > 128:
            zoom_factors = (128 / volume.shape[0],
                            128 / volume.shape[1],
                            128 / volume.shape[2])
            volume = ndimage.zoom(volume, zoom_factors, order=1)

        return volume

    except ImportError:
        print("⚠️ nibabel not installed. Install with: pip install nibabel")
        return None
    except Exception as e:
        print(f"❌ Error loading NIfTI file: {e}")
        return None
