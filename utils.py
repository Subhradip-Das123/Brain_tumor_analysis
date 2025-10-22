import numpy as np
import plotly.graph_objects as go
import cv2
from scipy import ndimage


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
    # Resize heatmap to match image dimensions
    heatmap_resized = cv2.resize(heatmap, (img_bgr.shape[1], img_bgr.shape[0]))

    # Normalize heatmap to 0-255 range
    heatmap_normalized = np.uint8(255 * heatmap_resized / heatmap_resized.max())

    # Apply colormap (JET is common for medical imaging)
    heatmap_colored = cv2.applyColorMap(heatmap_normalized, cv2.COLORMAP_JET)

    # Blend original image with heatmap
    overlay = cv2.addWeighted(img_bgr, 1 - alpha, heatmap_colored, alpha, 0)

    return overlay


def make_pseudo3d(img_bgr, gradcam_heatmap=None, depth=24):
    """
    Create an enhanced pseudo-3D MRI volume with improved depth representation.
    OPTIMIZED for web performance.

    Args:
        img_bgr: Input MRI image in BGR format
        gradcam_heatmap: Optional GradCAM heatmap to highlight tumor regions
        depth: Number of slices to generate (default 24 for performance)

    Returns:
        volume: 3D numpy array (depth, height, width)
        tumor_mask: 3D boolean array indicating tumor regions (if gradcam provided)
    """
    # Convert to grayscale and resize to smaller dimensions for performance
    img_gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    img_resized = cv2.resize(img_gray, (64, 64))  # Reduced from 128 to 64

    # Normalize to 0-1 range
    img_normalized = img_resized.astype(np.float32) / 255.0

    # Create volume with improved depth simulation
    volume = []
    tumor_mask = []

    center_slice = depth // 2

    for i in range(depth):
        # Calculate distance from center
        distance_from_center = abs(i - center_slice) / center_slice

        # Apply Gaussian blur that increases with distance from center
        blur_amount = int(1 + distance_from_center * 3)  # Reduced blur
        if blur_amount % 2 == 0:
            blur_amount += 1

        blurred = cv2.GaussianBlur(img_normalized, (blur_amount, blur_amount), 0)

        # Apply intensity falloff for depth effect
        intensity_factor = 1.0 - (distance_from_center * 0.3)
        slice_img = blurred * intensity_factor

        volume.append(slice_img)

        # Process tumor mask if GradCAM provided
        if gradcam_heatmap is not None:
            # Resize heatmap to match volume dimensions
            heatmap_resized = cv2.resize(gradcam_heatmap, (64, 64))  # Match volume size
            heatmap_normalized = heatmap_resized / (heatmap_resized.max() + 1e-8)

            # Create tumor region (threshold at 0.5)
            tumor_slice = heatmap_normalized > 0.5

            # Tumor appears in middle slices primarily
            if abs(i - center_slice) < depth // 3:
                tumor_mask.append(tumor_slice)
            else:
                tumor_mask.append(np.zeros_like(tumor_slice, dtype=bool))
        else:
            tumor_mask.append(np.zeros((64, 64), dtype=bool))

    volume = np.array(volume)
    tumor_mask = np.array(tumor_mask)

    return volume, tumor_mask


def volume_to_html(volume, tumor_mask=None, title="3D MRI Brain Visualization"):
    """
    Convert 3D MRI volume into an interactive Plotly 3D visualization.
    OPTIMIZED for web performance with reduced data points.

    Args:
        volume: 3D numpy array (depth, height, width)
        tumor_mask: Optional 3D boolean array for tumor regions
        title: Title for the visualization

    Returns:
        HTML string for embedding in web application
    """
    depth, height, width = volume.shape

    # CRITICAL: Subsample the data for performance
    # Take every 2nd point to reduce rendering load
    step = 2
    volume_sub = volume[::step, ::step, ::step]
    depth_sub, height_sub, width_sub = volume_sub.shape

    # Create coordinate meshgrid for subsampled data
    x = np.linspace(0, 1, width_sub)
    y = np.linspace(0, 1, height_sub)
    z = np.linspace(0, 1, depth_sub)

    X, Y, Z = np.meshgrid(x, y, z, indexing='ij')

    # Flatten arrays
    x_flat = X.flatten()
    y_flat = Y.flatten()
    z_flat = Z.flatten()
    values = volume_sub.flatten()

    # Create base brain volume
    fig = go.Figure()

    # Add main brain volume with optimized settings
    fig.add_trace(go.Volume(
        x=x_flat,
        y=y_flat,
        z=z_flat,
        value=values,
        isomin=0.15,
        isomax=0.85,
        opacity=0.1,  # Lower opacity for faster rendering
        surface_count=10,  # Reduced from 17
        colorscale='Gray',
        name='Brain Tissue',
        showscale=False,
        caps=dict(x_show=False, y_show=False, z_show=False),
        lighting=dict(ambient=0.5, diffuse=0.8, specular=0.2, roughness=0.5)
    ))

    # Add tumor region if provided (also subsampled)
    if tumor_mask is not None and np.any(tumor_mask):
        tumor_sub = tumor_mask[::step, ::step, ::step]
        tumor_values = tumor_sub.astype(float) * volume_sub
        tumor_flat = tumor_values.flatten()

        # Only show tumor regions with significant values
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
                surface_count=8,  # Reduced from 10
                colorscale='Hot',
                name='Tumor Region',
                showscale=True,
                colorbar=dict(title="Intensity", x=1.05, len=0.7)
            ))

    # Update layout for better visualization and performance
    fig.update_layout(
        title=dict(
            text=title,
            x=0.5,
            xanchor='center',
            font=dict(size=16)
        ),
        width=600,  # Reduced from 700
        height=600,
        margin=dict(l=0, r=0, b=0, t=40),
        scene=dict(
            xaxis=dict(
                title='',
                showticklabels=False,
                backgroundcolor='rgb(20, 20, 20)',
                gridcolor='rgb(50, 50, 50)',
                showbackground=True
            ),
            yaxis=dict(
                title='',
                showticklabels=False,
                backgroundcolor='rgb(20, 20, 20)',
                gridcolor='rgb(50, 50, 50)',
                showbackground=True
            ),
            zaxis=dict(
                title='',
                showticklabels=False,
                backgroundcolor='rgb(20, 20, 20)',
                gridcolor='rgb(50, 50, 50)',
                showbackground=True
            ),
            bgcolor='rgb(10, 10, 10)',
            camera=dict(
                eye=dict(x=1.5, y=1.5, z=1.3)
            )
        ),
        paper_bgcolor='rgb(30, 30, 30)',
        font=dict(color='white'),
        # Performance optimization
        uirevision='constant',
    )

    # Return lightweight HTML
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


def load_nifti_volume(nifti_path):
    """
    Load a NIfTI format 3D MRI volume.

    Args:
        nifti_path: Path to .nii or .nii.gz file

    Returns:
        volume: 3D numpy array
    """
    try:
        import nibabel as nib

        # Load NIfTI file
        nii_img = nib.load(nifti_path)
        volume = nii_img.get_fdata()

        # Normalize to 0-1 range
        volume = (volume - volume.min()) / (volume.max() - volume.min())

        # Resize if too large (for performance)
        if volume.shape[0] > 128:
            zoom_factors = (128 / volume.shape[0],
                            128 / volume.shape[1],
                            128 / volume.shape[2])
            volume = ndimage.zoom(volume, zoom_factors, order=1)

        return volume

    except ImportError:
        print("nibabel not installed. Install with: pip install nibabel")
        return None
    except Exception as e:
        print(f"Error loading NIfTI file: {e}")
        return None
