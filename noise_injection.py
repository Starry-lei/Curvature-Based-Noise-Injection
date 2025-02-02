import numpy as np
import open3d as o3d
from sklearn.neighbors import NearestNeighbors
import random
def add_curvature_based_noise(pcd, k=20, base_noise=0.01, scale_factor=10):
    """
    Adds noise to the point cloud based on local curvature.
    
    Parameters:
        pcd (open3d.geometry.PointCloud): The PCA-reconstructed point cloud.
        k (int): Number of nearest neighbors to use for curvature estimation.
        base_noise (float): Base noise amplitude.
        
    Returns:
        open3d.geometry.PointCloud: The point cloud with added noise.
    """
    # Convert points to numpy array
    points = np.asarray(pcd.points)
    nbrs = NearestNeighbors(n_neighbors=k, algorithm='auto').fit(points)
    _, indices = nbrs.kneighbors(points)
    
    # Initialize an array for modulated noise standard deviations
    noise_sigmas = np.zeros(points.shape[0])

    # print("show indices shape:",indices.shape)
    # (3072, 20)
    # exit()
    
    for i, neighbors in enumerate(indices):
        # Compute the covariance matrix for the local neighborhood
        local_pts = points[neighbors]
        cov = np.cov(local_pts.T)
        # Get eigenvalues (sorted in ascending order)
        eigenvalues, _ = np.linalg.eigh(cov)
        # Use the smallest eigenvalue as a proxy for local flatness (or inversely for curvature)
        curvature = eigenvalues[0]
        # print("curvature * scale_factor:", curvature * scale_factor)
        # Map curvature to noise amplitude; adjust the mapping function as needed
        # Higher curvature -> larger noise amplitude (here we use a simple proportional mapping)
        noise_sigmas[i] = base_noise * (1 + curvature * scale_factor)  # the factor 10 is arbitrary and can be tuned
    
    # Generate noise for each point
    noise = np.array([np.random.randn(3) * sigma for sigma in noise_sigmas])
    noisy_points = points + noise
    
    # Create new point cloud with noisy points
    pcd_noisy = o3d.geometry.PointCloud()
    pcd_noisy.points = o3d.utility.Vector3dVector(noisy_points)
    
    # Optionally copy colors or other attributes if available
    if pcd.has_colors():
        pcd_noisy.colors = pcd.colors
    
    return pcd_noisy

# Example usage:
# data_path= "data_demo/pca_recon_fcd25e25dfffff7af51f77a6d7299806.txt"
data_path ="data_demo/pca_recon_ff529b9ad2d5c6abf7e98086e1ca9511.txt"
data_demo= np.loadtxt(data_path)
data_demo= data_demo[:,0:3]
data_demo = data_demo[~np.all(data_demo == 0, axis=1)]

pcd = o3d.geometry.PointCloud()
pcd.points = o3d.utility.Vector3dVector(data_demo)
# vis
o3d.visualization.draw_geometries([pcd])
# exit()
# Assume pcd is your PCA-reconstructed (and possibly normalized) point cloud
noise_std_min=0.005
noise_std_max=0.015
noise_std = random.uniform(noise_std_min, noise_std_max)
print("noise_std:",noise_std)
pcd_noisy = add_curvature_based_noise(pcd, k=20, base_noise=noise_std, scale_factor=15000)
o3d.visualization.draw_geometries([pcd_noisy])
