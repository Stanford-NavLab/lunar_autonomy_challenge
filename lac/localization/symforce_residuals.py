import symforce.symbolic as sf
import symforce.typing as T

from lac.utils.frames import camera_to_opencv


def odometry_residual(
    world_T_a: sf.Pose3,
    world_T_b: sf.Pose3,
    a_T_b: sf.Pose3,
    diagonal_sigmas: sf.V6,
    epsilon: sf.Scalar,
) -> sf.V6:
    """
    Residual on the relative pose between two timesteps of the robot.
    Args:
        world_T_a: First pose in the world frame
        world_T_b: Second pose in the world frame
        a_T_b: Relative pose measurement between the poses
        diagonal_sigmas: Diagonal standard deviation of the tangent-space error
        epsilon: Small number for singularity handling
    """
    a_T_b_predicted = world_T_a.inverse() * world_T_b
    tangent_error = a_T_b_predicted.local_coordinates(a_T_b, epsilon=epsilon)
    return T.cast(sf.V6, sf.M.diag(diagonal_sigmas.to_flat_list()).inv() * sf.V6(tangent_error))


def imu_gyro_residual(
    T_curr: sf.Pose3,
    T_prev: sf.Pose3,
    angvel: sf.V3,
    dt: float,
    sigma: float,
) -> sf.V3:
    """
    Angular velocity constraint on 2 consecutive poses based on IMU equations.
    """
    R_curr = T_curr.R.to_rotation_matrix()
    R_prev = T_prev.R.to_rotation_matrix()
    rotmat_der = (R_curr - R_prev) / dt
    angvel_mat = rotmat_der * R_curr.T
    expected_angvel = sf.V3(angvel_mat[2, 1], angvel_mat[0, 2], angvel_mat[1, 0])
    return sf.V3(angvel - expected_angvel) / sigma


def imu_accel_residual(
    T_curr: sf.Pose3,
    T_prev: sf.Pose3,
    T_prev_prev: sf.Pose3,
    accel: sf.V3,
    gravity: sf.V3,
    dt: float,
    sigma: float,
) -> sf.V3:
    """
    Acceleration constraint on 3 consecutive poses based on IMU equations.
    """
    expected_accel = (T_curr.t + T_prev_prev.t - 2.0 * T_prev.t) / dt**2.0 + gravity
    expected_accel = T_curr.R.inverse() * expected_accel
    return sf.V3(accel - expected_accel) / sigma


def bearing_residual(
    robot_pose: sf.Pose3,
    landmark_pose: sf.Pose3,
    los_vector: sf.Vector3,
    sigma: sf.Scalar,
    epsilon: sf.Scalar,
) -> sf.V1:
    """
    Residual on the bearing measurement between the robot and a landmark.
    Args:
        robot_pose: Pose of the robot in the world frame
        landmark_pose: Pose of the landmark in the world frame
        los_vector: Unit vector pointing from the robot to the landmark
        sigma: Standard deviation of the bearing measurement
        epsilon: Small number for singularity handling
    """
    robot_to_landmark = (landmark_pose.t - robot_pose.t).normalized(epsilon=epsilon)
    angle_error = sf.acos_safe(robot_to_landmark.dot(los_vector), epsilon=epsilon)
    return sf.V1(angle_error / sigma)


def reprojection_residual(
    world_point: sf.V3,
    world_T_rover: sf.Pose3,
    rover_T_cam: sf.Pose3,
    pixel: sf.V2,
    camera_cal: sf.LinearCameraCal,
    sigma: float,
    epsilon: sf.Scalar,
) -> sf.V2:
    """
    Visual landmark reprojection residual.

    Args:
        point: 3D point in camera frame
        pose: Rover pose in world frame
        pixel: Pixel coordinates of the point in the image
        camera: Camera calibration
        sigma: Standard deviation of the residual
    """
    point_rover = world_T_rover.inverse() * world_point
    point_cam = rover_T_cam.inverse() * point_rover
    point_opencv = sf.V3(camera_to_opencv(point_cam))

    proj_pixel, is_valid = camera_cal.pixel_from_camera_point(point_opencv, epsilon=epsilon)
    return sf.V2(proj_pixel - pixel) / sigma
