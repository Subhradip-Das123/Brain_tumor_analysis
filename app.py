import streamlit as st
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import numpy as np
import cv2
import base64
import io
import imageio.v2 as imageio
from gradcam_torch import GradCAM
from utils import overlay_heatmap, make_pseudo3d, volume_to_html
import gdown
import os

# ------------------------------
# Load model safely (auto-download)
# ------------------------------
@st.cache_resource
def load_model():
    model_path = "model/brain_tumor_model.pth"
    if not os.path.exists(model_path):
        os.makedirs("model", exist_ok=True)
        file_id = "1qUajdKsWAvqU1L2uP2zsgZVtDwCJsiWi"  # ✅ Google Drive ID
        url = f"https://drive.google.com/uc?id={file_id}"
        gdown.download(url, model_path, quiet=False)

    model = models.vgg16(weights='IMAGENET1K_V1')
    model.classifier = nn.Sequential(
        nn.Flatten(),
        nn.Linear(25088, 128),
        nn.ReLU(),
        nn.Dropout(0.3),
        nn.Linear(128, 3),
        nn.Softmax(dim=1)
    )

    for module in model.modules():
        if isinstance(module, nn.ReLU):
            module.inplace = False

    model.load_state_dict(torch.load(model_path, map_location='cpu'))
    model.eval()
    return model


# ------------------------------
# Helper Functions
# ------------------------------
def image_to_datauri(img_bgr):
    buf = io.BytesIO()
    imageio.imwrite(buf, cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB), format='png')
    encoded = base64.b64encode(buf.getvalue()).decode()
    return "data:image/png;base64," + encoded


# ------------------------------
# Streamlit UI
# ------------------------------
st.set_page_config(page_title="🧠 Brain Tumor Detection", layout="wide")
st.title("🧠 Brain Tumor Detection using VGG16 + GradCAM")

st.write("Upload an MRI image to predict the tumor type and visualize GradCAM + 3D representation.")

uploaded_file = st.file_uploader("📤 Upload MRI Image", type=["jpg", "jpeg", "png"])

if uploaded_file:
    st.image(uploaded_file, caption="Uploaded MRI Image", use_container_width=True)
    st.info("Processing... Please wait ⏳")

    # Load and preprocess
    img_bytes = np.frombuffer(uploaded_file.read(), np.uint8)
    img_bgr = cv2.imdecode(img_bytes, cv2.IMREAD_COLOR)
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    img_pil = Image.fromarray(img_rgb)

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    input_tensor = transform(img_pil).unsqueeze(0)

    # Predict
    model = load_model()
    gradcam = GradCAM(model, target_layer="features.29")
    cam, class_idx = gradcam.generate_cam(input_tensor)

    class_names = ["Glioma", "Meningioma", "Pituitary"]
    label = class_names[class_idx]

    # Overlay GradCAM
    heatmap = overlay_heatmap(img_bgr, cam)

    # Create pseudo-3D view
    volume, tumor_mask = make_pseudo3d(img_bgr, gradcam_heatmap=cam, depth=20)
    vol_html = volume_to_html(volume, tumor_mask)

    # Display results
    st.success(f"### 🧩 Predicted Tumor Type: **{label}**")

    col1, col2 = st.columns(2)
    with col1:
        st.image(cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB), caption="Original MRI", use_container_width=True)
    with col2:
        st.image(cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB), caption="GradCAM Heatmap", use_container_width=True)

    st.markdown("### 🧠 3D Brain Visualization")
    st.components.v1.html(vol_html, height=650, scrolling=False)

