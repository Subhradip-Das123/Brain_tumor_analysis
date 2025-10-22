import streamlit as st
from PIL import Image
import torch
import cv2
import numpy as np
from torchvision import transforms
import os
import gdown
from gradcam_torch import GradCAM
from utils import load_model, preprocess_image, predict_tumor, overlay_heatmap, make_pseudo3d, volume_to_fig

# -----------------------------
# 🧠 Page Setup
# -----------------------------
st.set_page_config(page_title="🧠 Brain Tumor Detection", layout="wide")
st.title("🧠 Brain Tumor Detection using VGG16 + Grad-CAM + 3D Visualization")

# -----------------------------
# 📦 Auto-download Model from Drive if Missing
# -----------------------------
MODEL_PATH = "model/brain_tumor_model.pth"

if not os.path.exists(MODEL_PATH):
    os.makedirs("model", exist_ok=True)
    file_id = "1qUajdKsWAvqU1L2uP2zsgZVtDwCJsiWi"  # ✅ your Google Drive model ID
    url = f"https://drive.google.com/uc?export=download&id={file_id}"
    st.warning("📦 Downloading model file (~70MB)... please wait.")
    gdown.download(url, MODEL_PATH, quiet=False)
    st.success("✅ Model downloaded successfully!")

# -----------------------------
# 🧠 Load Model
# -----------------------------
@st.cache_resource
def get_model():
    return load_model(MODEL_PATH)

model = get_model()
target_layer = 'features.29'
st.success("✅ Model loaded successfully!")

# -----------------------------
# 📤 File Upload
# -----------------------------
uploaded_file = st.file_uploader("📤 Upload an MRI image (jpg/png)", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Read image
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    img_bgr = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    img_pil = Image.fromarray(img_rgb)
    st.image(img_rgb, caption="🩺 Uploaded MRI Image", use_container_width=True)

    # Preprocess and predict
    image_tensor = preprocess_image(img_pil)

    with st.spinner("🔍 Running model prediction..."):
        pred_idx, confidence = predict_tumor(model, image_tensor)
        class_names = ["Glioma", "Meningioma", "Pituitary"]
        label = class_names[pred_idx]
        st.markdown(f"### 🧩 **Predicted Tumor Type:** `{label}` ({confidence*100:.2f}% confidence)")

    # GradCAM heatmap
    with st.spinner("🔥 Generating Grad-CAM heatmap..."):
        gradcam = GradCAM(model, target_layer)
        cam, _ = gradcam.generate_cam(image_tensor)
        heatmap = overlay_heatmap(img_bgr, cam)

    # Display GradCAM results
    col1, col2 = st.columns(2)
    with col1:
        st.image(img_rgb, caption="🧠 Original MRI", use_container_width=True)
    with col2:
        st.image(heatmap, caption="🔥 Grad-CAM Overlay", use_container_width=True)

    # 3D visualization
    with st.spinner("🧩 Creating pseudo 3D MRI visualization..."):
        volume, tumor_mask = make_pseudo3d(img_bgr, gradcam_heatmap=cam, depth=24)
        fig = volume_to_fig(volume, tumor_mask)

    st.markdown("### 🧠 3D MRI Brain Visualization")
    st.plotly_chart(fig, use_container_width=True)
