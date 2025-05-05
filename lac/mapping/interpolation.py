import numpy as np
from scipy.interpolate import griddata, Rbf


def interpolate_heights(height_array: np.ndarray) -> np.ndarray:
    """Interpolate the heights of the height array using cubic interpolation.

    height_array: np.ndarray
        The height array with shape (N, N, 3) where the last dimension is (x, y, z).

    """
    N = height_array.shape[0]
    valid_mask = height_array[..., 2] != -np.inf
    x_known = height_array[..., 0][valid_mask]
    y_known = height_array[..., 1][valid_mask]
    z_known = height_array[..., 2][valid_mask]

    # Extract all (x, y) grid points
    x_all = height_array[..., 0].ravel()
    y_all = height_array[..., 1].ravel()

    # Cubic interpolation
    z_interp = griddata((x_known, y_known), z_known, (x_all, y_all), method="cubic")

    # Fill NaNs using nearest neighbor
    z_nn = griddata((x_known, y_known), z_known, (x_all, y_all), method="nearest")
    z_interp[np.isnan(z_interp)] = z_nn[np.isnan(z_interp)]

    height_array_interpolated = height_array.copy()
    height_array_interpolated[..., 0] = x_all.reshape(N, N)
    height_array_interpolated[..., 1] = y_all.reshape(N, N)
    height_array_interpolated[..., 2] = z_interp.reshape(N, N)
    return height_array_interpolated


def interpolate_heights_rbf(
    height_array: np.ndarray, function: str = "multiquadric", smooth: float = 0.5
) -> np.ndarray:
    """
    Interpolate the heights of the height array using RBF interpolation.

    NOTE: really slow, didn't finish after 4 minutes

    Parameters:
    - height_array: np.ndarray
        Array of shape (N, N, 3) where the last dimension is (x, y, z).
    - function: str
        Type of RBF kernel. Options: 'multiquadric', 'inverse', 'gaussian', 'linear', 'cubic', 'quintic', 'thin_plate'
    - smooth: float
        Smoothing parameter for the RBF interpolator.

    Returns:
    - height_array_interpolated: np.ndarray
        The input array with z-values interpolated over missing cells.
    """
    N = height_array.shape[0]
    valid_mask = height_array[..., 2] != -np.inf

    # Extract known (x, y, z) values
    x_known = height_array[..., 0][valid_mask]
    y_known = height_array[..., 1][valid_mask]
    z_known = height_array[..., 2][valid_mask]

    # Create interpolator
    rbf = Rbf(x_known, y_known, z_known, function=function, smooth=smooth)

    # Evaluate on full grid
    x_all = height_array[..., 0].ravel()
    y_all = height_array[..., 1].ravel()
    z_interp = rbf(x_all, y_all)

    # Replace z-values
    height_array_interpolated = height_array.copy()
    height_array_interpolated[..., 2] = z_interp.reshape(N, N)

    return height_array_interpolated
