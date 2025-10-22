import numpy as np
import cv2
import plotly.graph_objects as go
from scipy import ndimage


# ✅ Overlay GradCAM Heatmap on MRI Image
def overlay_heatmap(img_bgr, heatmap, alpha=0.4):
    """
    Overlay a heatmap (e.g., GradCAM) on the original MRI image.
    Args:
        img_bgr: Original image (BGR)
        heatmap: Heatmap array (grayscale, values 0–255)
        alpha: Transparency factor for overlay (0–1)
    """
    heatmap_resized = cv2.resize(heatmap, (img_bgr.shape[1], img_bgr.shape[0]))
    heatmap_norm = np.uint8(255 * heatmap_resized / (heatmap_resized.max() + 1e-8))
    heatmap_color = cv2.applyColorMap(heatmap_norm, cv2.COLORMAP_JET)
    blended = cv2.addWeighted(img_bgr, 1 - alpha, heatmap_color, alpha, 0)
    return blended


# ✅ Generate Pseudo-3D MRI Volume
def make_pseudo3d(img_bgr, gradcam_heatmap=None, depth=24):
    """
    Create a pseudo-3D MRI volume visualization for Streamlit web app.
    """
    img_gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    img_resized = cv2.resize(img_gray, (64, 64))
    img_norm = img_resized.astype(np.float32) / 255.0

    volume, tumor_mask = [], []
    center = depth // 2

    for i in range(depth):
        dist = abs(i - center) / center
        blur_k = int(1 + dist * 3)
        if blur_k % 2 == 0:
            blur_k += 1
        blurred = cv2.GaussianBlur(img_norm, (blur_k, blur_k), 0)
        slice_img = blurred * (1.0 - dist * 0.3)
        volume.append(slice_img)

        if gradcam_heatmap is not None:
            heat_resized = cv2.resize(gradcam_heatmap, (64, 64))
            heat_norm = heat_resized / (heat_resized.max() + 1e-8)
            tumor_slice = heat_norm > 0.5
            if abs(i - center) < depth // 3:
                tumor_mask.append(tumor_slice)
            else:
                tumor_mask.append(np.zeros_like(tumor_slice, dtype=bool))
        else:
            tumor_mask.append(np.zeros((64, 64), dtype=bool))

    return np.array(volume), np.array(tumor_mask)


# ✅ 3D Plotly Visualization (Optimized for Streamlit)
def volume_to_html(volume, tumor_mask=None, title="3D MRI Brain Visualization"):
    """
    Convert 3D MRI volume into an interactive Plotly 3D visualization.
    """
    d, h, w = volume.shape
    step = 2
    volume_sub = volume[::step, ::step, ::step]
    d_s, h_s, w_s = volume_sub.shape

    x = np.linspace(0, 1, w_s)
    y = np.linspace(0, 1, h_s)
    z = np.linspace(0, 1, d_s)
    X, Y, Z = np.meshgrid(x, y, z, indexing='ij')

    fig = go.Figure()

    # Base brain volume
    fig.add_trace(go.Volume(
        x=X.flatten(),
        y=Y.flatten(),
        z=Z.flatten(),
        value=volume_sub.flatten(),
        isomin=0.15, isomax=0.85,
        opacity=0.1,
        surface_count=10,
        colorscale='Gray',
        showscale=False
    ))

    # Tumor region
    if tumor_mask is not None and np.any(tumor_mask):
        tumor_sub = tumor_mask[::step, ::step, ::step]
        tumor_values = tumor_sub.astype(float) * volume_sub
        mask = tumor_values.flatten() > 0.1
        fig.add_trace(go.Volume(
            x=X.flatten()[mask],
            y=Y.flatten()[mask],
            z=Z.flatten()[mask],
            value=tumor_values.flatten()[mask],
            isomin=0.1,
            isomax=1.0,
            opacity=0.4,
            surface_count=8,
            colorscale='Hot',
            name='Tumor Region',
            showscale=True
        ))

    fig.update_layout(
        title=dict(text=title, x=0.5, xanchor='center'),
        width=600,
        height=600,
        margin=dict(l=0, r=0, b=0, t=40),
        paper_bgcolor='rgb(30,30,30)',
        font=dict(color='white')
    )

    return fig.to_html(full_html=False, include_plotlyjs='cdn')


# ✅ Load NIfTI 3D MRI (Optional for future extension)
def load_nifti_volume(nifti_path):
    """Load .nii or .nii.gz MRI volume."""
    try:
        import nibabel as nib
        nii = nib.load(nifti_path)
        vol = nii.get_fdata()
        vol = (vol - vol.min()) / (vol.max() - vol.min())
        if vol.shape[0] > 128:
            zoom_factors = (128 / vol.shape[0], 128 / vol.shape[1], 128 / vol.shape[2])
            vol = ndimage.zoom(vol, zoom_factors, order=1)
        return vol
    except ImportError:
        print("Install nibabel for NIfTI support: pip install nibabel")
        return None
    except Exception as e:
        print(f"Error loading NIfTI: {e}")
        return None
