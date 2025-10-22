import streamlit as st
from PIL import Image
import torch, cv2, numpy as np
from torchvision import transforms, models
from gradcam_torch import GradCAM
from utils import overlay_heatmap, make_pseudo3d, volume_to_html
import base64, io, os

# -----------------------------
# 1️⃣ Page Configuration
# -----------------------------
st.set_page_config(page_title="🧠 Brain Tumor Detection", layout="wide")
st.title("🧠 Brain Tumor Detection using Grad-CAM & 3D MRI Visualization")

# -----------------------------
# 2️⃣ Load Model
# -----------------------------
import torch.nn as nn

@st.cache_resource
def load_model():
    model = models.vgg16(weights='IMAGENET1K_V1')
    model.classifier = nn.Sequential(
        nn.Flatten(),
        nn.Linear(25088, 128),
        nn.ReLU(),
        nn.Dropout(0.3),
        nn.Linear(128, 3),
        nn.Softmax(dim=1)
    )

    # Disable in-place ReLU for GradCAM
    for module in model.modules():
        if isinstance(module, nn.ReLU):
            module.inplace = False

    model.load_state_dict(torch.load("model/brain_tumor_model.pth", map_location='cpu'))
    model.eval()
    return model

model = load_model()
target_layer = 'features.29'  # GradCAM layer for VGG16
st.success("✅ Model loaded successfully!")

# -----------------------------
# 3️⃣ Image Preprocessing
# -----------------------------
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

def image_to_datauri(img_bgr):
    """Convert BGR image to base64-encoded HTML for display"""
    _, buf = cv2.imencode('.png', img_bgr)
    encoded = base64.b64encode(buf).decode()
    return f"data:image/png;base64,{encoded}"

# -----------------------------
# 4️⃣ File Upload
# -----------------------------
uploaded_file = st.file_uploader("📤 Upload an MRI image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    img_bgr = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    img_pil = Image.fromarray(img_rgb)
    input_tensor = transform(img_pil).unsqueeze(0)

    st.image(img_rgb, caption="🩺 Uploaded MRI Image", use_container_width=True)

    with st.spinner("🧠 Generating Grad-CAM visualization..."):
        gradcam = GradCAM(model, target_layer)
        cam, class_idx = gradcam.generate_cam(input_tensor)

        class_names = ["Glioma", "Meningioma", "Pituitary"]
        label = class_names[class_idx]

        heatmap = overlay_heatmap(img_bgr, cam)
        volume, tumor_mask = make_pseudo3d(img_bgr, gradcam_heatmap=cam, depth=20)
        vol_html = volume_to_html(volume, tumor_mask)

    st.markdown(f"### 🧩 **Predicted Tumor Type:** `{label}`")
    col1, col2 = st.columns(2)

    with col1:
        st.image(img_bgr, caption="🩺 Original MRI", use_container_width=True)
    with col2:
        st.image(heatmap, caption="🔥 Grad-CAM Overlay", use_container_width=True)

    st.markdown("### 🧠 3D MRI Visualization")
    st.components.v1.html(vol_html, height=650, scrolling=False)

