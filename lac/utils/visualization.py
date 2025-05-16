"""Visualization utils"""

import numpy as np
import cv2
import matplotlib.pyplot as plt
import matplotlib.colors

from lac.slam.semantic_feature_tracker import TrackedPoints


def int_to_color(i, cmap_name="tab20", hex=False):
    cmap = plt.get_cmap(cmap_name)
    color = cmap(i % cmap.N)
    if hex:
        return matplotlib.colors.rgb2hex(color[:3])  # Plotly uses hex colors
    else:
        return tuple(int(c * 255) for c in color[:3])


def cv_display_text(text, height=300, width=500, font_scale=1, color=(255, 255, 255), thickness=2):
    """Clear the image and display new text."""
    window_name = "Output Window"
    img = np.zeros((height, width, 3), dtype=np.uint8)
    cv2.putText(img, text, (50, 150), cv2.FONT_HERSHEY_SIMPLEX, font_scale, color, thickness)
    cv2.imshow(window_name, img)


def color_mask(mask: np.ndarray, color) -> np.ndarray:
    """Color a mask with a given color.

    mask : np.ndarray (H, W, 3) - Binary mask
    color : tuple (3) - RGB color

    """
    # Ensure mask has 3 channels
    if len(mask.shape) == 2:
        mask = 255 * cv2.cvtColor(mask.astype(np.uint8), cv2.COLOR_GRAY2RGB)
    mask = mask * np.array(color)
    return mask


def overlay_mask(image_gray, mask, color=(1, 0, 0)):
    """
    image_gray : np.ndarray (H, W) - grayscale image
    mask : np.ndarray (H, W) - Binary mask
    color : tuple (3) - RGB color
    """
    image_rgb = cv2.cvtColor(image_gray, cv2.COLOR_GRAY2BGR)
    mask_colored = color_mask(mask, color).astype(image_rgb.dtype)
    return cv2.addWeighted(image_rgb, 1.0, mask_colored, beta=0.5, gamma=0)


def overlay_stereo_rock_depths(left_image, depth_results):
    """
    left_image : np.ndarray (H, W, 3) - RGB image
    depth_results : list of dict - Stereo depth results (output of stereo_depth_from_segmentation)
    """
    overlay = left_image.copy()
    text_offset = np.array([5, -5])
    for result in depth_results:
        left_centroid = result["left_centroid"]
        depth = result["depth"]
        cv2.circle(overlay, tuple(left_centroid), 5, (0, 255, 0), -1)
        cv2.putText(
            overlay,
            f"{depth:.2f} m",
            tuple(left_centroid + text_offset),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 255, 0),
            2,
        )
    return overlay


def overlay_points(image, points, color=(0, 255, 0), size=5, thickness=-1):
    """
    left_image : np.ndarray (H, W, 3) - RGB image
    points : np.ndarray (N, 2) - points to overlay
    color : color in BGR
    """
    overlay = image.copy()
    for point in points:
        cv2.circle(overlay, tuple(point.astype(int)), size, color, thickness)
    return overlay


def overlay_tracked_points(image: np.ndarray, tracks: TrackedPoints):
    if len(image.shape) == 2:
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    new_kps = tracks.points[tracks.lengths == 0]
    tracked_kps = tracks.points[tracks.lengths != 0]
    overlay = overlay_points(image, new_kps, color=(0, 165, 255), size=3)  # orange
    overlay = overlay_points(overlay, tracked_kps, color=(0, 255, 0), size=3)  # green
    return overlay


def overlay_tag_detections(image_gray, detections):
    """
    image_gray : np.ndarray (H, W) - grayscale image
    detections : list of apriltag.Detection - AprilTag detections
    """
    image_rgb = cv2.cvtColor(image_gray, cv2.COLOR_GRAY2BGR)
    overlay = image_rgb.copy()
    for detection in detections:
        for pt in detection.corners:
            cv2.circle(overlay, tuple(pt.astype(int)), 5, (0, 255, 0), -1)
    return overlay


def draw_steering_arc(image, steering, l=0.4, color=(0, 0, 255), thickness=3):
    """
    Overlays an arc on the input image showing the predicted trajectory of a rover.

    Parameters:
      image: Input image (BGR numpy array).
      steering: Steering value in radians. Zero means straight ahead.
      l: Look-ahead distance as a fraction of image height.
      color: Color for the arc (default red in BGR).
      thickness: Thickness of the drawn arc.

    Returns:
      The image with the arc overlay.
    """
    # overlay = image.copy()
    overlay = np.zeros_like(image, dtype=np.uint8).copy()
    h, w = image.shape[:2]
    # Define the vehicle's (or camera's) location as the bottom-center of the image.
    cx, cy = w // 2, h

    # Determine the arc length in pixels (look-ahead distance)
    arc_length = l * h

    # If steering is nearly zero, draw a straight line upward.
    if abs(steering) < 1e-3:
        end_point = (cx, int(cy - arc_length))
        cv2.line(overlay, (cx, cy), end_point, color, thickness)
    else:
        # For nonzero steering, we will compute a circular arc.
        # We choose R so that an angular sweep Δθ = arc_length/R corresponds approximately to our look-ahead.
        R = arc_length / np.tan(abs(steering))

        # For a left turn (steering > 0), the turning circle is centered to the left.
        if steering > 0:
            circle_center = (cx - int(R), cy)
            # We want the rover's position (cx,cy) to lie on the circle.
            # In our circle parameterization, we use the standard parametric form:
            #    P(theta) = center + (R*cos(theta), R*sin(theta))
            # For a left turn let theta=0 correspond to P = (cx,cy):
            theta_start = 0.0
            # The angular sweep (in radians) that gives an arc of length arc_length is:
            sweep = arc_length / R
            # For a left turn we traverse the circle “clockwise” (i.e. decreasing theta)
            theta_end = theta_start - sweep
        else:
            # For a right turn (steering < 0) the turning circle is centered to the right.
            circle_center = (cx + int(R), cy)
            # Now the point (cx,cy) will correspond to theta = π:
            theta_start = np.pi
            sweep = arc_length / R
            # For a right turn we traverse the circle “clockwise” (which, starting at π, is increasing)
            theta_end = theta_start + sweep

        # Generate a set of points along the circular arc.
        num_points = 50  # increase for a smoother arc
        thetas = np.linspace(theta_start, theta_end, num_points)
        pts = []
        # Make sure to work with floats
        cx_center, cy_center = float(circle_center[0]), float(circle_center[1])
        for theta in thetas:
            # Parametric equation for a circle:
            x = cx_center + R * np.cos(theta)
            y = cy_center + R * np.sin(theta)
            pts.append((int(x), int(y)))
        pts = np.array(pts, np.int32).reshape((-1, 1, 2))
        cv2.polylines(overlay, [pts], isClosed=False, color=color, thickness=thickness)

    # Blend the overlay with the original image (here 50% transparency for the arc)
    output = cv2.addWeighted(image, 1.0, overlay, 0.5, 0)
    return output


# def image_grid(
#     images,
#     rows=None,
#     cols=None,
#     fill: bool = True,
#     show_axes: bool = False,
#     rgb: bool = True,
# ):
#     """
#     A util function for plotting a grid of images.

#     Args:
#         images: (N, H, W, 4) array of RGBA images
#         rows: number of rows in the grid
#         cols: number of columns in the grid
#         fill: boolean indicating if the space between images should be filled
#         show_axes: boolean indicating if the axes of the plots should be visible
#         rgb: boolean, If True, only RGB channels are plotted.
#             If False, only the alpha channel is plotted.

#     Returns:
#         None
#     """
#     if (rows is None) != (cols is None):
#         raise ValueError("Specify either both rows and cols or neither.")

#     if rows is None:
#         rows = len(images)
#         cols = 1

#     gridspec_kw = {"wspace": 0.0, "hspace": 0.0} if fill else {}
#     fig, axarr = plt.subplots(rows, cols, gridspec_kw=gridspec_kw, figsize=(15, 9))
#     bleed = 0
#     fig.subplots_adjust(left=bleed, bottom=bleed, right=(1 - bleed), top=(1 - bleed))


#     for ax, im in zip(axarr.ravel(), images):
#         if rgb:
#             # only render RGB channels
#             ax.imshow(im[..., :3])
#         else:
#             # only render Alpha channel
#             ax.imshow(im[..., 3])
#         if not show_axes:
#             ax.set_axis_off()
def image_grid(
    images,
    rows=None,
    cols=None,
    fill: bool = True,
    show_axes: bool = False,
    rgb: bool = True,
    figsize=(15, 9),
):
    """
    A utility function for plotting a grid of images.

    Args:
        images: List or array of images with shapes (H, W), (H, W, 3), or (H, W, 4).
        rows: Number of rows in the grid.
        cols: Number of columns in the grid.
        fill: Boolean indicating if the space between images should be filled.
        show_axes: Boolean indicating if the axes of the plots should be visible.
        rgb: Boolean, If True, only RGB channels are plotted.
            If False, only the alpha channel is plotted (for RGBA images).

    Returns:
        None
    """
    if (rows is None) != (cols is None):
        raise ValueError("Specify either both rows and cols or neither.")

    if rows is None:
        rows = len(images)
        cols = 1

    gridspec_kw = {"wspace": 0.0, "hspace": 0.0} if fill else {}
    fig, axarr = plt.subplots(rows, cols, gridspec_kw=gridspec_kw, figsize=figsize)

    axarr = np.array(axarr).reshape(-1)  # Flatten in case of 2D grid

    for ax, im in zip(axarr, images):
        if im.ndim == 2:
            # Grayscale image (H, W)
            ax.imshow(im, cmap="gray")
        elif im.shape[-1] == 3:
            # RGB image (H, W, 3)
            ax.imshow(im)
        elif im.shape[-1] == 4:
            # RGBA image (H, W, 4)
            if rgb:
                ax.imshow(im[..., :3])  # Only RGB channels
            else:
                ax.imshow(im[..., 3], cmap="gray")  # Only Alpha channel
        else:
            raise ValueError(f"Unsupported image shape: {im.shape}")

        if not show_axes:
            ax.set_axis_off()

    plt.show()
