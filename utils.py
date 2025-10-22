import torch
import torch.nn as nn     # ✅ ADD THIS LINE
import os
import gdown
import torchvision.transforms as transforms
from PIL import Image
import numpy as np
import cv2
import plotly.graph_objects as go
from scipy import ndimage

# ======================================================
# 1️⃣ Load Model
# ======================================================
# Define your CNN architecture (must match training model)
def load_model():
    """Load trained VGG16 model and weights from Google Drive"""
    model_path = "model/brain_tumor_model.pth"

    # 1️⃣ Download model if missing
    if not os.path.exists(model_path):
        os.makedirs("model", exist_ok=True)
        file_id = "1qUajdKsWAvqU1L2uP2zsgZVtDwCJsiWi"  # ✅ your Google Drive file ID
        url = f"https://drive.google.com/uc?export=download&id={file_id}"
        gdown.download(url, model_path, quiet=False)

    # 2️⃣ Initialize the same architecture used during training
    model = models.vgg16(pretrained=False)
    model.classifier[6] = nn.Linear(4096, 2)  # 2 output classes: Tumor / No Tumor

    # 3️⃣ Load the saved state_dict
    state_dict = torch.load(model_path, map_location=torch.device('cpu'))
    model.load_state_dict(state_dict)

    # 4️⃣ Set to evaluation mode
    model.eval()
    return model
# ======================================================
# 2️⃣ Preprocess and Predict
# ======================================================
def preprocess_image(image):
    """Apply same preprocessing used during training"""
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5], std=[0.5])  # adjust if trained differently
    ])
    return transform(image).unsqueeze(0)


def predict_tumor(model, image_tensor):
    """Run inference on image and return prediction"""
    with torch.no_grad():
        outputs = model(image_tensor)
        _, predicted = torch.max(outputs.data, 1)
        probs = torch.nn.functional.softmax(outputs, dim=1)[0]
        return predicted.item(), probs[predicted].item()


# ======================================================
# 3️⃣ Grad-CAM Overlay Function
# ======================================================
def overlay_heatmap(img_bgr, heatmap, alpha=0.4):
    """
    Overlay a heatmap (e.g., GradCAM) on the original MRI image.

    Args:
        img_bgr: Original image in BGR format
        heatmap: Heatmap array (grayscale, values 0-255)
        alpha: Transparency factor for overlay (0.0-1.0)

    Returns:
        Blended image with heatmap overlay
    """
    heatmap_resized = cv2.resize(heatmap, (img_bgr.shape[1], img_bgr.shape[0]))
    heatmap_normalized = np.uint8(255 * heatmap_resized / (heatmap_resized.max() + 1e-8))
    heatmap_colored = cv2.applyColorMap(heatmap_normalized, cv2.COLORMAP_JET)
    overlay = cv2.addWeighted(img_bgr, 1 - alpha, heatmap_colored, alpha, 0)
    return overlay


# ======================================================
# 4️⃣ Pseudo-3D MRI Volume Generator
# ======================================================
def make_pseudo3d(img_bgr, gradcam_heatmap=None, depth=24):
    """
    Create a pseudo-3D MRI volume for web visualization.

    Args:
        img_bgr: Input MRI image in BGR format
        gradcam_heatmap: Optional GradCAM heatmap
        depth: Number of slices (default 24)

    Returns:
        volume, tumor_mask
    """
    img_gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    img_resized = cv2.resize(img_gray, (64, 64))
    img_normalized = img_resized.astype(np.float32) / 255.0

    volume, tumor_mask = [], []
    center_slice = depth // 2

    for i in range(depth):
        distance_from_center = abs(i - center_slice) / center_slice
        blur_amount = int(1 + distance_from_center * 3)
        if blur_amount % 2 == 0:
            blur_amount += 1
        blurred = cv2.GaussianBlur(img_normalized, (blur_amount, blur_amount), 0)
        intensity_factor = 1.0 - (distance_from_center * 0.3)
        slice_img = blurred * intensity_factor
        volume.append(slice_img)

        if gradcam_heatmap is not None:
            heatmap_resized = cv2.resize(gradcam_heatmap, (64, 64))
            heatmap_normalized = heatmap_resized / (heatmap_resized.max() + 1e-8)
            tumor_slice = heatmap_normalized > 0.5
            if abs(i - center_slice) < depth // 3:
                tumor_mask.append(tumor_slice)
            else:
                tumor_mask.append(np.zeros_like(tumor_slice, dtype=bool))
        else:
            tumor_mask.append(np.zeros((64, 64), dtype=bool))

    return np.array(volume), np.array(tumor_mask)


# ======================================================
# 5️⃣ 3D Visualization (Plotly)
# ======================================================
def volume_to_html(volume, tumor_mask=None, title="3D MRI Brain Visualization"):
    """
    Convert 3D MRI volume into an interactive Plotly visualization.
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

    # Base brain structure
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
                isomin=0.1,
                isomax=1.0,
                opacity=0.4,
                surface_count=8,
                colorscale='Hot',
                name='Tumor Region',
                showscale=True,
                colorbar=dict(title="Intensity", x=1.05, len=0.7)
            ))

    fig.update_layout(
        title=dict(text=title, x=0.5, xanchor='center', font=dict(size=16)),
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
# 6️⃣ Load 3D NIfTI Volume
# ======================================================
def load_nifti_volume(nifti_path):
    """Load a NIfTI format 3D MRI volume (.nii or .nii.gz)."""
    try:
        import nibabel as nib
        nii_img = nib.load(nifti_path)
        volume = nii_img.get_fdata()
        volume = (volume - volume.min()) / (volume.max() - volume.min())
        if volume.shape[0] > 128:
            zoom_factors = (128 / volume.shape[0], 128 / volume.shape[1], 128 / volume.shape[2])
            volume = ndimage.zoom(volume, zoom_factors, order=1)
        return volume
    except ImportError:
        print("⚠️ nibabel not installed. Install with: pip install nibabel")
        return None
    except Exception as e:
        print(f"Error loading NIfTI file: {e}")
        return None





