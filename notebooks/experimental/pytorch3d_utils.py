import torch
from pytorch3d.structures import Meshes


def structured_grid_to_pytorch3d_mesh(grid):
    """
    Convert an NxNx3 structured grid into a triangular mesh in PyTorch3D.

    Args:
        grid (np.ndarray): Shape (N, N, 3), regular 2.5D grid of (x, y, z) points.

    Returns:
        Meshes: A PyTorch3D mesh object.
    """
    N, M, _ = grid.shape  # N x M grid of (x,y,z) points
    vertices = grid.reshape(-1, 3)  # Flatten into (N*M, 3)

    # Generate face indices for a regular grid of quads split into triangles
    faces = []
    for i in range(N - 1):
        for j in range(M - 1):
            # Get indices in the flattened array
            v0 = i * M + j
            v1 = i * M + (j + 1)
            v2 = (i + 1) * M + j
            v3 = (i + 1) * M + (j + 1)

            # Each quad is split into two triangles
            faces.append([v0, v1, v2])  # Lower-left triangle
            faces.append([v1, v3, v2])  # Upper-right triangle

    # Convert to PyTorch tensors
    verts = torch.tensor(vertices, dtype=torch.float32)
    faces = torch.tensor(faces, dtype=torch.int64)

    # Create PyTorch3D Mesh
    mesh = Meshes(verts=[verts], faces=[faces])
    return mesh
