import numpy as np
import laspy
import warnings
import argparse
from sklearn.linear_model import RANSACRegressor
from sklearn.mixture import GaussianMixture
from scipy.stats import entropy

"""
Note:
The terrain normalization part (remove_terrain) is included for demonstration purposes only.
It may not be suitable for all types of terrain or LiDAR data.
"""

def remove_terrain(points, grid_size=1.0, use_gmm=True, n_components=1):
    """
    Extract ground points using grid lowest points and normalize terrain.
    Optionally use GMM instead of RANSAC for ground modeling.

    Args:
        points (np.ndarray): Point cloud data, shape (N, 3), columns (x, y, z).
        grid_size (float): Grid size (same unit as points) for ground extraction.
        use_gmm (bool): Whether to use GMM for ground modeling instead of RANSAC.
        n_components (int): Number of Gaussian components for GMM (default 1).

    Returns:
        np.ndarray: Normalized point cloud, shape (N, 3), columns (x, y, z_norm).
    """
    x, y, z = points[:, 0], points[:, 1], points[:, 2]

    # Create grid indices
    x_idx = np.floor((x - np.min(x)) / grid_size).astype(int)
    y_idx = np.floor((y - np.min(y)) / grid_size).astype(int)

    # Group points by grid and select the lowest point in each grid
    ground_points = {}
    for xi, yi, xi_val, yi_val, zi_val in zip(x_idx, y_idx, x, y, z):
        key = (xi, yi)
        if key not in ground_points or zi_val < ground_points[key][2]:
            ground_points[key] = (xi_val, yi_val, zi_val)

    ground_array = np.array(list(ground_points.values()))
    xy_ground = ground_array[:, :2]
    z_ground = ground_array[:, 2]

    if use_gmm:
        # Use GMM to model ground surface
        gmm = GaussianMixture(n_components=n_components, covariance_type='full', random_state=0)
        gmm.fit(np.column_stack((xy_ground, z_ground)))

        # Predict expected Z for each (x, y) based on the GMM
        xy_all = np.column_stack((x, y))
        # Append dummy Z=0 for prediction, only XY is meaningful
        xy_dummy_z = np.hstack((xy_all, np.zeros((xy_all.shape[0], 1))))
        z_pred = np.zeros(xy_all.shape[0])
        responsibilities = gmm.predict_proba(xy_dummy_z)
        means = gmm.means_[:, 2]  # Mean Z of each component
        z_pred = responsibilities @ means
    else:
        # Use RANSAC to model ground surface
        ransac = RANSACRegressor()
        ransac.fit(xy_ground, z_ground)
        z_pred = ransac.predict(np.column_stack((x, y)))

    # Normalize Z values
    z_norm = z - z_pred
    return np.column_stack((x, y, z_norm))


def read_las(file_path):
    """
    Read a LAS file and extract point cloud data.

    Args:
        file_path (str): Path to the LAS file.

    Returns:
        np.ndarray: Point cloud data of shape (N, 3).
    """
    las = laspy.read(file_path)
    points = np.vstack((las.x, las.y, las.z)).T
    return points

def split_into_windows_z(points, window_size_x, window_size_y, z_box, rh=10, ratio=1):
    """
    Split the point cloud into sub-windows.

    Args:
        points (np.ndarray): Input point cloud of shape (N, 3).
        window_size_x (float): Window size along the x-axis.
        window_size_y (float): Window size along the y-axis.
        z_box (int): Number of divisions along the z-axis.
        rh (float, optional): Total height along z-axis (default 10).
        ratio (float, optional): Step-to-window size ratio (default 1 for no overlap).

    Returns:
        tuple: (list of sub-windows, number of windows)
    """
    windows = []
    x_min, y_min, z_min = np.min(points[:, 0]), np.min(points[:, 1]), np.min(points[:, 2])
    x_max, y_max, z_max = np.max(points[:, 0]), np.max(points[:, 1]), np.max(points[:, 2])

    height_unit = rh / z_box

    for x_start in np.arange(x_min, x_max, window_size_x * ratio):
        for y_start in np.arange(y_min, y_max, window_size_y * ratio):
            subwindows = []
            for z_height in range(z_box):
                x_end = x_start + window_size_x
                y_end = y_start + window_size_y
                z_start = z_min + z_height * height_unit
                z_end = z_min + (z_height + 1) * height_unit

                window_points = points[
                    (points[:, 0] >= x_start) & (points[:, 0] < x_end) &
                    (points[:, 1] >= y_start) & (points[:, 1] < y_end) &
                    (points[:, 2] >= z_start) & (points[:, 2] < z_end)
                ]
                subwindows.append(window_points)
            if subwindows:
                windows.append(subwindows)

    return windows, len(windows)

def compute_3d_histogram_distribution(points, bins):
    """
    Compute and normalize the 3D histogram as a probability distribution.

    Args:
        points (np.ndarray): Point cloud data of shape (N, 3).
        bins (int): Number of histogram bins per dimension.

    Returns:
        np.ndarray: Normalized 3D histogram as a probability distribution.
    """
    if len(points) == 0:
        hist = np.full((bins, bins, bins), 1e-43)
        return hist / hist.sum()

    hist, _ = np.histogramdd(
        points,
        bins=bins,
        range=[[np.min(points[:, 0]), np.max(points[:, 0])],
               [np.min(points[:, 1]), np.max(points[:, 1])],
               [np.min(points[:, 2]), np.max(points[:, 2])]]
    )
    hist = hist / hist.sum()
    hist += 1e-43
    return hist

def kl_divergence(p, q):
    """
    Compute the KL divergence between two distributions.

    Args:
        p (np.ndarray): First distribution.
        q (np.ndarray): Second distribution.

    Returns:
        float: KL divergence value.
    """
    epsilon = 1e-43
    p = np.clip(p, epsilon, None)
    q = np.clip(q, epsilon, None)
    return entropy(p.flatten(), q.flatten())

def process_las(file, ws, zbox, ratio=1, normalize_terrain=False, grid_size=1.0,use_gmm=False, n_components=1):
    """
    Process a LAS file by splitting into windows and computing the VSRE score.

    Args:
        file (str): Path to the LAS file.
        ws (float): Window size.
        zbox (int): Number of divisions along z-axis.
        ratio (float, optional): Step-to-window size ratio (default 1 for no overlap).
        normalize_terrain (bool, optional): Whether to perform terrain normalization (default False).
        grid_size (float, optional): Grid size for ground extraction during normalization (default 1.0).
        use_gmm (bool): Use GMM for ground modeling if True.
        n_components (int): Number of components for GMM.

    Returns:
        float: Average KL divergence across all windows.
    """
    points = read_las(file)

    if normalize_terrain:
        points = remove_terrain(points, grid_size=grid_size, use_gmm=use_gmm,
        n_components=n_components)

    windows, sz = split_into_windows_z(points, window_size_x=ws, window_size_y=ws, z_box=zbox, ratio=ratio)

    lsZ = []
    for subwindows in windows:
        kl_values_per_z = []
        distributions = [compute_3d_histogram_distribution(window, bins=20) for window in subwindows]

        for i in range(len(subwindows)):
            for j in range(len(subwindows)):
                if i != j:
                    kl_values_per_z.append(kl_divergence(distributions[i], distributions[j]))

        if len(subwindows) > 1:
            lsZ.append(np.sum(kl_values_per_z) / (len(subwindows) * (len(subwindows) - 1)))
        else:
            lsZ.append(np.sum(kl_values_per_z) / (len(subwindows) * len(subwindows)))
            warnings.warn("Only one valid window was calculated.") 

    return sum(lsZ) / sz

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Compute the VSRE score of a LAS file after optional terrain normalization and window splitting.")
    
    parser.add_argument("--file", type=str, required=True, help="Path to input LAS file.")
    parser.add_argument("--ws", type=float, default=10.0, help="Window size (default: 10.0m).")
    parser.add_argument(
    "--zbox", type=int, default=20, help="Number of divisions along z-axis (default: 20). (Each window size along z-axis = rh divided by zbox.)")
    parser.add_argument("--ratio", type=float, default=1.0, help="Step-to-window size ratio (default: 1.0, no overlap).")
    parser.add_argument("--normalize_terrain", action='store_true', help="Whether to normalize terrain before processing.")
    parser.add_argument("--grid_size", type=float, default=1.0, help="Grid size for ground extraction during normalization (default: 1.0).")
    parser.add_argument("--use_gmm", action='store_true', help="Use GMM instead of RANSAC for ground modeling.")
    parser.add_argument("--n_components", type=int, default=1, help="Number of GMM components if GMM is used (default: 1).")
    
    args = parser.parse_args()

    VSRE_score = process_las(
        file=args.file,
        ws=args.ws,
        zbox=args.zbox,
        ratio=args.ratio,
        normalize_terrain=args.normalize_terrain,
        grid_size=args.grid_size,
        use_gmm=args.use_gmm,
        n_components=args.n_components
    )

    print(f"Vegetation Structure Relative Entropy: {VSRE_score:.6f}")
