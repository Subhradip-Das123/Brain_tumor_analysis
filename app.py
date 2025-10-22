from flask import Flask, render_template, request
from PIL import Image
import torch, cv2, numpy as np
from torchvision import transforms
from gradcam_torch import GradCAM
from utils import overlay_heatmap, make_pseudo3d, volume_to_html
import base64, io
import imageio.v2 as imageio
import os

app = Flask(__name__)

# ---- Load trained model ----
import torch.nn as nn
from torchvision import models

model = models.vgg16(weights='IMAGENET1K_V1')
model.classifier = nn.Sequential(
    nn.Flatten(),
    nn.Linear(25088, 128),
    nn.ReLU(),
    nn.Dropout(0.3),
    nn.Linear(128, 3),
    nn.Softmax(dim=1)
)

# Disable in-place ReLU for Grad-CAM compatibility
for module in model.modules():
    if isinstance(module, nn.ReLU):
        module.inplace = False

model.load_state_dict(torch.load("model/brain_tumor_model.pth", map_location='cpu'))
model.eval()


print("✅ Model loaded successfully (VGG16)")

target_layer = 'features.29'  # ✅ Correct Grad-CAM layer for VGG16

# ---- Preprocessing ----
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

def image_to_datauri(img_bgr):
    buf = io.BytesIO()
    imageio.imwrite(buf, cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB), format='png')
    encoded = base64.b64encode(buf.getvalue()).decode()
    return "data:image/png;base64," + encoded

@app.route('/', methods=['GET','POST'])
def index():
    if request.method == 'POST':
        file = request.files['image']
        img_bytes = np.frombuffer(file.read(), np.uint8)
        img_bgr = cv2.imdecode(img_bytes, cv2.IMREAD_COLOR)

        # preprocess for model
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        img_pil = Image.fromarray(img_rgb)
        input_tensor = transform(img_pil).unsqueeze(0)

        gradcam = GradCAM(model, target_layer)
        cam, class_idx = gradcam.generate_cam(input_tensor)

        # prediction label
        class_names = ["Glioma", "Meningioma", "Pituitary"]
        label = class_names[class_idx]

        # overlay Grad-CAM
        heatmap = overlay_heatmap(img_bgr, cam)

        # pseudo-3D view
        volume, tumor_mask = make_pseudo3d(img_bgr, gradcam_heatmap=cam, depth=20)  # Reduced from 32
        vol_html = volume_to_html(volume, tumor_mask)

        return render_template('result.html',
                               label=label,
                               original=image_to_datauri(img_bgr),
                               heatmap=image_to_datauri(heatmap),
                               vol_html=vol_html)

    return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True)


