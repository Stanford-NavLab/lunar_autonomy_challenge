import numpy as np
import carla
import math
import cv2


from PIL import Image


def get_camera_intrinsics(camera_key, fov_deg=70):
    """
    Returns (f_x, f_y, c_x, c_y) given resolution W x H and a horizontal FoV in degrees.
    """
    W, H = int(camera_key["width"]), int(camera_key["height"])

    fov_rad = math.radians(fov_deg)  

    f_x = W / (2.0 * math.tan(fov_rad / 2.0))
    f_y = f_x # square pixels
    
    c_x = W / 2.0
    c_y = H / 2.0
    
    K = np.array([[f_x, 0, c_x],
                  [0, f_y, c_y],
                  [0, 0, 1]])
    
    D = np.zeros((5,1))  

    return K, D



def get_homogenous_transform(transform):
    """
    Convert LAC transform (location + pitch,roll,yaw) into a 4x4 homogeneous matrix.
    rotation order: yaw about Z, pitch about Y, roll about X (right-handed).
    """
    x, y, z = transform.location.x, transform.location.y, transform.location.z
    pitch, roll, yaw = transform.rotation.pitch, transform.rotation.roll, transform.rotation.yaw

    # Sine/cosine
    cp, sp = math.cos(pitch), math.sin(pitch)
    cr, sr = math.cos(roll),  math.sin(roll)
    cy, sy = math.cos(yaw),   math.sin(yaw)

    # LAC: +yaw => rotation about Z, +pitch => about Y, +roll => about X (all clockwise from top)
    # But note the sign conventions. Usually CARLA uses a left-handed approach for angles, but LAC says right-handed.
    # (Double-check or adapt if sign mismatch arises.)

    Rz = np.array([[ cy, -sy,  0],
                   [ sy,  cy,  0],
                   [  0,   0,  1]])
    Ry = np.array([[ cp,  0,  sp],
                   [  0,  1,   0],
                   [-sp,  0,  cp]])
    Rx = np.array([[ 1,   0,    0],
                   [ 0,  cr,  -sr],
                   [ 0,  sr,   cr]])

    R = Rz @ Ry @ Rx  # Adjust ?

    # 4x4 homogenous transform matrix
    M = np.eye(4)
    M[0:3, 0:3] = R
    M[0:3, 3] = (x,y,z)


    return M


def get_extrinsic_left_to_right(self, left_key, right_key):
   
    left_tf = self.get_camera_position(left_key)
    right_tf = self.get_camera_position(right_key)

    M_left2rover = get_homogenous_transform(left_tf)   
    M_right2rover  = get_homogenous_transform(right_tf) 

    
    M_rover2left = np.linalg.inv(M_left2rover)
    M_left2right = M_rover2left @ M_right2rover

    return M_left2right


def decompose_homogenous_transform(M):
    R = M[0:3, 0:3]
    t = M[0:3, 3]
    return R, t


def stereo_rectify(left_img,right_img, K_left, D_left, K_right, D_right, R, t):
    """
    R,t are from left->right (3x3 rotation, 3x1 translation).
    image_size is (W, H).
    Returns the stereo rectification parameters for each camera.
    """
    
    img_size = left_img.shape[::-1] 

    R1, R2, P1, P2, Q, _, _ = cv2.stereoRectify(
        cameraMatrix1=K_left, distCoeffs1=D_left,
        cameraMatrix2=K_right, distCoeffs2=D_right,
        imageSize=img_size,
        R=R, T=t,
        flags=cv2.CALIB_ZERO_DISPARITY,
        alpha=0
    )

    mapxL, mapyL = cv2.initUndistortRectifyMap(K_left, D_left, R1, P1, img_size, cv2.CV_32FC1)
    mapxR, mapyR = cv2.initUndistortRectifyMap(K_right, D_right, R2, P2, img_size, cv2.CV_32FC1)

    
    left_rect  = cv2.remap(left_img,  mapxL, mapyL, cv2.INTER_LINEAR)
    right_rect = cv2.remap(right_img, mapxR, mapyR, cv2.INTER_LINEAR)
    
    return left_rect, right_rect

def color_mask(mask: np.ndarray, color) -> np.ndarray:
    """Color a mask with a given color.

    mask : np.ndarray (H, W, 3) - Binary mask
    color : tuple (3) - RGB color

    """
    mask = mask * np.array(color)
    return mask

def mask_centroid(mask: np.ndarray) -> tuple:
    """Compute the centroid of a binary mask."""
    M = cv2.moments(mask)
    cx = int(M["m10"] / M["m00"])
    cy = int(M["m01"] / M["m00"])
    return cx, cy

def segment_image(self, rectified_image):
    rectified_image_PIL = Image.fromarray(rectified_image).convert("RGB")
    results, full_mask = self.segmentation.segment_rocks(rectified_image_PIL)
    rectified_image_PIL_rgb = cv2.cvtColor(rectified_image, cv2.COLOR_GRAY2BGR)
    mask_colored = color_mask(full_mask, (0, 0, 1)).astype(rectified_image_PIL_rgb.dtype)

    max_area = 0
    max_mask = None
    for mask in results["masks"]:
        mask_area = np.sum(mask)
        if mask_area > max_area and mask_area < 50000:  # Filter out outliers
            max_area = mask_area
            max_mask = mask

    cx, cy = None, None
    if max_mask is not None and max_area > 1000:
        max_mask = max_mask.astype(np.uint8)
        cx, cy = mask_centroid(max_mask)
        #x, y, w, h = cv2.boundingRect(max_mask)
    
    return cx, cy

    