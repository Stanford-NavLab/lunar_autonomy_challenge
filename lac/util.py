import numpy as np
from scipy.spatial.transform import Rotation
from PIL import Image
import cv2 as cv


def transform_to_numpy(transform):
    """Convert a Transform object to a 4x4 pose matrix.

    The resulting pose matrix has +X forward, +Y left, +Z up.

    """
    t = np.array([transform.location.x, transform.location.y, transform.location.z])
    euler_angles = np.array(
        [transform.rotation.roll, transform.rotation.pitch, transform.rotation.yaw]
    )
    R = Rotation.from_euler("xyz", euler_angles).as_matrix()
    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = t
    return T


def to_blender_convention(pose):
    """Convert a camera pose matrix to Blender convention.

    The camera pose matrix is assumed to have above starting convention (+X forward, +Y left, +Z up)
    The Blender convention has -Z forward, +X right, +Y up.

    """
    R = pose[:3, :3]
    t = pose[:3, 3]
    # Convert the rotation matrix to the Blender convention
    rx, ry, rz = R[:, 0], R[:, 1], R[:, 2]
    R_blender = np.array([-ry, rz, -rx])
    return np.block([[R_blender, t[:, None]], [0, 0, 0, 1]])


def pose_to_rpy_pos(pose):
    """Convert a camera pose matrix to LAC convention.

    The camera pose matrix is assumed to have above starting convention (+X forward, +Y left, +Z up)
    The LAC convention has +X forward, +Y left, +Z up.

    """
    R = pose[:3, :3]
    t = pose[:3, 3]

    # Calculate yaw, pitch, roll using scipy Rotation
    r = Rotation.from_matrix(R)
    roll, pitch, yaw = r.as_euler("xyz")

    pos = np.array([t[0], t[1], t[2]])
    rpy = np.array([roll, pitch, yaw])

    return rpy, pos


def skew_symmetric(v):
    """Convert a 3D vector to a skew-symmetric matrix."""
    return np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])


def normalize_rotation_matrix(R):
    """Normalizes a rotation matrix using SVD."""
    U, _, Vt = np.linalg.svd(R)  # Singular Value Decomposition
    R_normalized = U @ Vt  # Reconstruct a valid rotation matrix
    return R_normalized


def rmse(predictions, targets):
    return np.sqrt(((predictions - targets) ** 2).mean())


def wrap_angle(angle):
    """Wrap an angle in radians to [-pi, pi]."""
    return (angle + np.pi) % (2 * np.pi) - np.pi


def np_img_to_PIL_rgb(img_array):
    """Convert a numpy image array to a PIL image."""
    return Image.fromarray(img_array).convert("RGB")


def color_mask(mask: np.ndarray, color) -> np.ndarray:
    """Color a mask with a given color.

    mask : np.ndarray (H, W, 3) - Binary mask
    color : tuple (3) - RGB color

    """
    mask = mask * np.array(color)
    return mask


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
    overlay = image.copy()
    h, w = image.shape[:2]
    # Define the vehicle's (or camera's) location as the bottom-center of the image.
    cx, cy = w // 2, h

    # Determine the arc length in pixels (look-ahead distance)
    arc_length = l * h

    # If steering is nearly zero, draw a straight line upward.
    if abs(steering) < 1e-3:
        end_point = (cx, int(cy - arc_length))
        cv.line(overlay, (cx, cy), end_point, color, thickness)
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
        cv.polylines(overlay, [pts], isClosed=False, color=color, thickness=thickness)

    # Blend the overlay with the original image (here 50% transparency for the arc)
    output = cv.addWeighted(image, 1.0, overlay, 0.5, 0)
    return output
