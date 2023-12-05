import torch
import numpy as np

def angle_between_matrices(mat1, mat2):
    # Ensure both matrices have the same shape
    assert mat1.shape == mat2.shape, "Matrices must be of the same shape"

    # Calculate the dot products (element-wise multiplication and sum across columns)
    dot_products = torch.sum(mat1 * mat2, dim=1)

    # Calculate the norms of the rows
    norms_mat1 = torch.norm(mat1, p=2, dim=1)
    norms_mat2 = torch.norm(mat2, p=2, dim=1)

    # Calculate the cosine of the angles
    cos_angles = dot_products / (norms_mat1 * norms_mat2)

    # Calculate the angles in radians
    angles_rad = torch.acos(cos_angles.clamp(-1, 1))  # Clamp for numerical stability

    # Convert angles to degrees
    angles_deg = angles_rad * 180 / torch.pi

    angles_deg = angles_deg.detach().cpu().numpy()
    reval = np.mean(angles_deg)

    return reval

def sigmoid_derivative(x):
    x = x.detach().numpy()
    return np.multiply(x, 1-x)