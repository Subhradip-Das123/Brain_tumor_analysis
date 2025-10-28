import numpy as np
import plotly.graph_objects as go
import cv2
from scipy import ndimage

# ==============================================================
# ✅ OVERLAY HEATMAP ON ORIGINAL IMAGE
# ==============================================================

def overlay_heatmap(img_bgr, heatmap, alpha=0.4):
    """
    Overlay a Grad-CAM heatmap on the original MRI image.
    Handles blank heatmaps gracefully (e.g., for 'No Tumor' cases).

    Args:
        img_bgr: Original image in BGR format
        heatmap: Grad-CAM heatmap (float32 array, values 0–1)
        alpha: Transparency factor for overlay (0.0–1.0)

    Returns:
        overlay: Image with heatmap blended
    """
    # Ensure valid heatmap
    if heatmap is None or np.all(heatmap == 0):
        return img_bgr  # No tumor highlight → return original image

    # Resize heatmap to match original
    heatmap_resized = cv2.resize(heatmap, (img_bgr.shape[1], img_bgr.shape[0]))

    # Normalize to 0–255 safely
    heatmap_normalized = np.uint8(255 * heatmap_resized / (heatmap_resized.max() + 1e-8))

    # Apply JET colormap for medical interpretability
    heatmap_colored = cv2.applyColorMap(heatmap_normalized, cv2.COLORMAP_JET)

    # Blend image and heatmap
    overlay = cv2.addWeighted(img_bgr, 1 - alpha, heatmap_colored, alpha, 0)
    return overlay


# ==============================================================
# ✅ GENERATE PSEUDO 3D MRI VISUALIZATION
# ==============================================================

def make_pseudo3d(img_bgr, gradcam_heatmap=None, depth=24):
    """
    Generate a pseudo-3D MRI volume highlighting tumor regions.
    Optimized for browser performance.

    Args:
        img_bgr: Input MRI image (BGR)
        gradcam_heatmap: Grad-CAM heatmap (float array)
        depth: Number of pseudo slices

    Returns:
        volume: Simulated 3D grayscale MRI stack
        tumor_mask: Boolean mask of tumor region
    """
    # Convert image to grayscale
    img_gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    img_resized = cv2.resize(img_gray, (64, 64))  # Downsample for speed

    img_normalized = img_resized.astype(np.float32) / 255.0

    volume = []
    tumor_mask = []
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

        # ---- Tumor Mask Processing ----
        if gradcam_heatmap is not None and np.any(gradcam_heatmap > 0):
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


# ==============================================================
# ✅ CONVERT 3D VOLUME TO HTML VISUALIZATION
# ==============================================================

def volume_to_html(volume, tumor_mask=None, title="3D MRI Brain Visualization"):
    """
    Render the pseudo-3D MRI + tumor mask as an interactive Plotly volume.

    Args:
        volume: 3D numpy array
        tumor_mask: Optional tumor mask
        title: HTML title string

    Returns:
        HTML string for embedding
    """
    depth, height, width = volume.shape
    step = 2  # Subsampling step for performance

    # Subsample
    volume_sub = volume[::step, ::step, ::step]
    depth_sub, height_sub, width_sub = volume_sub.shape

    x = np.linspace(0, 1, width_sub)
    y = np.linspace(0, 1, height_sub)
    z = np.linspace(0, 1, depth_sub)
    X, Y, Z = np.meshgrid(x, y, z, indexing='ij')

    x_flat = X.flatten()
    y_flat = Y.flatten()
    z_flat = Z.flatten()
    values = volume_sub.flatten()

    fig = go.Figure()

    # Base Brain Volume
    fig.add_trace(go.Volume(
        x=x_flat,
        y=y_flat,
        z=z_flat,
        value=values,
        isomin=0.15,
        isomax=0.85,
        opacity=0.1,
        surface_count=10,
        colorscale='Gray',
        name='Brain Tissue',
        showscale=False,
        caps=dict(x_show=False, y_show=False, z_show=False),
        lighting=dict(ambient=0.5, diffuse=0.8, specular=0.2, roughness=0.5)
    ))

    # Tumor Region Overlay (optional)
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
        scene=dict(
            xaxis=dict(showticklabels=False, backgroundcolor='rgb(20,20,20)'),
            yaxis=dict(showticklabels=False, backgroundcolor='rgb(20,20,20)'),
            zaxis=dict(showticklabels=False, backgroundcolor='rgb(20,20,20)'),
            bgcolor='rgb(10,10,10)',
            camera=dict(eye=dict(x=1.5, y=1.5, z=1.3))
        ),
        paper_bgcolor='rgb(30,30,30)',
        font=dict(color='white'),
        uirevision='constant'
    )

    return fig.to_html(
        full_html=False,
        include_plotlyjs='cdn',
        config={
            'displayModeBar': True,
            'displaylogo': False,
            'modeBarButtonsToRemove': ['toImage', 'sendDataToCloud'],
            'responsive': True
        }
    )


# ==============================================================
# ✅ LOAD NIFTI (.nii / .nii.gz) MRI FILES
# ==============================================================

def load_nifti_volume(nifti_path):
    """
    Load NIfTI (.nii / .nii.gz) 3D MRI files safely.

    Returns:
        volume: normalized 3D array or None
    """
    try:
        import nibabel as nib
        nii_img = nib.load(nifti_path)
        volume = nii_img.get_fdata()

        # Normalize 0–1
        volume = (volume - volume.min()) / (volume.max() - volume.min())

        # Resize large volumes
        if volume.shape[0] > 128:
            zoom_factors = (128 / volume.shape[0],
                            128 / volume.shape[1],
                            128 / volume.shape[2])
            volume = ndimage.zoom(volume, zoom_factors, order=1)

        return volume

    except ImportError:
        print("⚠️ nibabel not installed. Run: pip install nibabel")
        return None
    except Exception as e:
        print(f"❌ Error loading NIfTI file: {e}")
        return None
