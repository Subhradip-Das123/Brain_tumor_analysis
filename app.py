import os
import cv2
import gdown
import numpy as np
import streamlit as st
from PIL import Image

from utils import (
    load_model_tf,
    preprocess_image,
    predict_tumor,
    gradcam_heatmap,
    overlay_heatmap,
    make_pseudo3d,
    volume_to_fig,
)

# ----------------------------------------------------------
# Page config
# ----------------------------------------------------------
st.set_page_config(page_title="🧠 Brain Tumor Detection (Keras)", layout="wide")
st.title("🧠 Brain Tumor Detection (VGG16 • Keras) with Grad-CAM + 3D Visualization")

# ----------------------------------------------------------
# Auto-download model from Google Drive if missing
# ----------------------------------------------------------
MODEL_PATH = "model/brain_tumor_model.h5"
FILE_ID = "1qUajdKsWAvqU1L2uP2zsgZVtDwCJsiWi"  # <-- your Drive file id

if not os.path.exists(MODEL_PATH):
    os.makedirs("model", exist_ok=True)
    url = f"https://drive.google.com/uc?export=download&id={FILE_ID}"
    st.warning("📦 Downloading Keras model (~70MB) from Google Drive…")
    gdown.download(url, MODEL_PATH, quiet=False)
    st.success("✅ Model downloaded")

# ----------------------------------------------------------
# Load model (cached)
# ----------------------------------------------------------
@st.cache_resource
def get_model():
    return load_model_tf(MODEL_PATH)

model = get_model()
st.sidebar.success("✅ Model loaded")

# Optional: class names (adjust if needed)
CLASS_NAMES = ["Glioma", "Meningioma", "Pituitary"]

# ----------------------------------------------------------
# Upload image
# ----------------------------------------------------------
uploaded = st.file_uploader("📤 Upload an MRI image (jpg/png)", type=["jpg", "jpeg", "png"])

if uploaded:
    # Read image
    file_bytes = np.asarray(bytearray(uploaded.read()), dtype=np.uint8)
    img_bgr = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    img_pil = Image.fromarray(img_rgb)

    st.image(img_rgb, caption="🩺 Uploaded MRI", use_container_width=True)

    # Preprocess + predict
    with st.spinner("🔍 Running model prediction…"):
        x = preprocess_image(img_pil)  # shape (1, H, W, 3)
        pred_idx, confidence, probs = predict_tumor(model, x)
        label = CLASS_NAMES[pred_idx] if pred_idx < len(CLASS_NAMES) else f"Class {pred_idx}"
        st.markdown(f"### 🧩 **Prediction:** `{label}`  ·  Confidence: `{confidence*100:.2f}%`")

    # Grad-CAM
    with st.spinner("🔥 Generating Grad-CAM heatmap…"):
        cam = gradcam_heatmap(model, x)            # (H, W) float [0..1]
        heat_overlay = overlay_heatmap(img_bgr, cam)

    c1, c2 = st.columns(2)
    with c1:
        st.image(img_rgb, caption="🧠 Original MRI", use_container_width=True)
    with c2:
        st.image(heat_overlay, caption="🔥 Grad-CAM Overlay", use_container_width=True)

    # Pseudo-3D viz
    with st.spinner("🧩 Building pseudo 3D visualization…"):
        volume, tumor_mask = make_pseudo3d(img_bgr, gradcam_heatmap=cam, depth=24)
        fig = volume_to_fig(volume, tumor_mask)

    st.markdown("### 🧠 3D MRI Brain Visualization")
    st.plotly_chart(fig, use_container_width=True)
else:
    st.info("Upload an MRI image to get a prediction, Grad-CAM heatmap, and 3D visualization.")
