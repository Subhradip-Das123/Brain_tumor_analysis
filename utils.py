import os
import cv2
import numpy as np
import plotly.graph_objects as go
from PIL import Image
from scipy import ndimage

import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras import Model

# If you trained at IMAGE_SIZE=128, keep that here:
IMAGE_SIZE = 128  # change to your training size if different


# ======================================================
# 1) Load Keras model
# ======================================================
def load_model_tf(model_path="model/brain_tumor_model.h5"):
    """
    Load a Keras (.h5) model saved via model.save(...).
    """
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model not found at {model_path}")
    model = load_model(model_path)
    return model


# ======================================================
# 2) Preprocess and predict
# ======================================================
def preprocess_image(image: Image.Image):
    """
    Resize to training size and scale to [0,1] (match your training code).
    Returns a batch tensor (1, H, W, 3).
    """
    img = image.resize((IMAGE_SIZE, IMAGE_SIZE))
    arr = np.asarray(img).astype(np.float32) / 255.0
    if arr.ndim == 2:  # grayscale safety
        arr = np.stack([arr]*3, axis=-1)
    if arr.shape[-1] == 4:  # RGBA -> RGB
        arr = arr[..., :3]
    return np.expand_dims(arr, axis=0)


def predict_tumor(model, x):
    """
    Returns (pred_idx, confidence, probs array).
    """
    probs = model.predict(x, verbose=0)[0]  # (C,)
    pred_idx = int(np.argmax(probs))
    confidence = float(np.max(probs))
    return pred_idx, confidence, probs


# ======================================================
# 3) Grad-CAM for Keras
# ======================================================
def _find_last_conv_layer(model):
    """
    Find the last conv layer. If you used VGG16(include_top=False) inside Sequential,
    it’s typically model.get_layer('vgg16').get_layer('block5_conv3').
    This function searches robustly.
    """
    # Try nested VGG16 first
    try:
        vgg = model.get_layer('vgg16')
        for lname in ['block5_conv3', 'block5_conv2', 'block5_conv1']:
            try:
                return vgg.get_layer(lname).name
            except Exception:
                pass
    except Exception:
        pass

    # Generic fallback: last conv layer anywhere
    for layer in reversed(model.layers):
        try:
            if 'conv' in layer.name and len(layer.output_shape) == 4:
                return layer.name
        except Exception:
            # Some layers might not have output_shape before build
            continue

    raise ValueError("No suitable conv layer found for Grad-CAM.")


def gradcam_heatmap(model, x, eps=1e-8):
    """
    Compute Grad-CAM heatmap (H, W) float in [0,1] for the predicted class.
    """
    # 1) Find target conv layer
    last_conv_name = _find_last_conv_layer(model)
    last_conv_layer = model.get_layer('vgg16').get_layer(last_conv_name) if 'vgg16' in [l.name for l in model.layers] else model.get_layer(last_conv_name)

    # 2) Build a model that maps input -> (conv outputs, predictions)
    grad_model = Model(
        inputs=model.inputs,
        outputs=[last_conv_layer.output, model.output],
        name="gradcam_model",
    )

    # 3) Forward + gradient tape
    with tf.GradientTape() as tape:
        conv_outputs, preds = grad_model(x, training=False)
        pred_idx = tf.argmax(preds[0])
        loss = preds[:, pred_idx]

    # 4) Gradients of the top predicted class wrt conv outputs
    grads = tape.gradient(loss, conv_outputs)  # shape (1, Hc, Wc, C)

    # 5) Global-average-pool the gradients over width/height
    weights = tf.reduce_mean(grads, axis=(1, 2))  # (1, C)

    # 6) Weighted sum over channels to get CAM
    cam = tf.reduce_sum(tf.multiply(conv_outputs, tf.expand_dims(tf.expand_dims(weights, 1), 1)), axis=-1)  # (1, Hc, Wc)
    cam = cam[0].numpy()

    # 7) ReLU and normalize to [0,1]
    cam = np.maximum(cam, 0)
    cam = cam / (cam.max() + eps)

    # 8) Resize to input image size used by model
    H, W = x.shape[1], x.shape[2]
    cam = cv2.resize(cam, (W, H), interpolation=cv2.INTER_CUBIC)
    return cam


# ======================================================
# 4) Visualization helpers
# ======================================================
def overlay_heatmap(img_bgr, heatmap, alpha=0.4):
    """
    Blend OpenCV BGR image with a JET heatmap built from `heatmap` (H, W) in [0,1].
    """
    heatmap_uint8 = np.uint8(255 * heatmap)
    heatmap_color = cv2.applyColorMap(heatmap_uint8, cv2.COLORMAP_JET)
    overlay = cv2.addWeighted(img_bgr, 1 - alpha, heatmap_color, alpha, 0)
    return overlay


def make_pseudo3d(img_bgr, gradcam_heatmap=None, depth=24):
    """
    Produce a simple 3D volume (D, H, W) using blurred/intensity-attenuated copies
    of a 2D MRI, and optional tumor mask from Grad-CAM.
    """
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
            tumor_mask.append(tumor if abs(i - center) < depth // 3 else np.zeros_like(tumor, dtype=bool))
        else:
            tumor_mask.append(np.zeros((64, 64), dtype=bool))

    return np.array(volume), np.array(tumor_mask)


def volume_to_fig(volume, tumor_mask=None, title="3D MRI Brain Visualization"):
    """
    Convert 3D volume (D,H,W) to a Plotly Volume figure for Streamlit.
    """
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
        colorscale='Gray', showscale=False,
    ))

    if tumor_mask is not None and np.any(tumor_mask):
        t_sub = tumor_mask[::step, ::step, ::step]
        t_vals = t_sub.astype(float) * vol_sub
        t_flat = t_vals.flatten()
        idx = t_flat > 0.1
        if np.any(idx):
            fig.add_trace(go.Volume(
                x=X.flatten()[idx], y=Y.flatten()[idx], z=Z.flatten()[idx],
                value=t_flat[idx],
                isomin=0.1, isomax=1.0,
                opacity=0.4, surface_count=8,
                colorscale='Hot', showscale=True, name="Tumor",
            ))

    fig.update_layout(
        title=dict(text=title, x=0.5),
        width=800, height=700,
        scene=dict(aspectmode="cube"),
        margin=dict(l=0, r=0, b=0, t=40),
        paper_bgcolor='rgb(30,30,30)',
        font=dict(color='white'),
    )
    return fig

