import streamlit as st
from PIL import Image
import numpy as np
import cv2
from utils import load_model, preprocess_image, predict_tumor, overlay_heatmap, make_pseudo3d, volume_to_html

# ----------------------------------------------------------
# Streamlit App Configuration
# ----------------------------------------------------------
st.set_page_config(page_title="🧠 Brain Tumor Detection", layout="centered")

st.title("🧠 Brain Tumor Detection using MRI (PyTorch)")
st.markdown("Upload an MRI scan to detect the presence of a brain tumor and visualize its region using Grad-CAM and 3D rendering.")

# ----------------------------------------------------------
# Load Model (Automatically from Google Drive if missing)
# ----------------------------------------------------------
with st.spinner("Loading model..."):
    model = load_model()
st.sidebar.success("✅ Model loaded successfully")

# ----------------------------------------------------------
# Upload MRI Image
# ----------------------------------------------------------
uploaded_file = st.file_uploader("📤 Upload an MRI image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Read and display uploaded image
    image = Image.open(uploaded_file).convert('RGB')
    st.image(image, caption="Uploaded MRI Image", use_column_width=True)

    # ----------------------------------------------------------
    # Model Prediction
    # ----------------------------------------------------------
    image_tensor = preprocess_image(image)
    pred_class, confidence = predict_tumor(model, image_tensor)

    if pred_class == 1:
        st.error(f"🚨 Tumor Detected! Confidence: {confidence*100:.2f}%")
    else:
        st.success(f"✅ No Tumor Detected. Confidence: {confidence*100:.2f}%")

    # ----------------------------------------------------------
    # Grad-CAM + Pseudo-3D Visualization
    # ----------------------------------------------------------
    with st.spinner("Generating 3D visualization..."):
        img_bgr = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)

        # Simulated Grad-CAM heatmap (placeholder)
        heatmap = np.random.rand(224, 224).astype(np.float32)
        overlay_img = overlay_heatmap(img_bgr, heatmap)

        st.image(overlay_img, caption="Grad-CAM Overlay", use_column_width=True)

        # Create pseudo-3D MRI volume
        volume, tumor_mask = make_pseudo3d(img_bgr, heatmap)
        html_plot = volume_to_html(volume, tumor_mask)

        # Display interactive 3D visualization
        st.components.v1.html(html_plot, height=650, scrolling=False)

st.caption("Developed by **Subhradip Das** | Powered by PyTorch & Streamlit")

