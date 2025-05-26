import torch
import torch.nn.functional as F
import torch_cluster  # make sure it's installed

def compute_pointcloud_normals(xyz, k=20):
    """
    Computes normals for a point cloud using PCA on k-NN neighbors.
    Args:
        xyz: (N, 3) tensor of point positions
        k: number of neighbors
    Returns:
        normals: (N, 3) unit normals
    """
    idx = torch_cluster.knn_graph(xyz, k=k, loop=False)  # (2, E)
    row, col = idx
    N = xyz.shape[0]

    normals = torch.zeros_like(xyz)

    for i in range(N):
        neighbor_idxs = col[row == i]
        if neighbor_idxs.shape[0] >= 3:
            nb = xyz[neighbor_idxs] - xyz[i]
            cov = nb.T @ nb
            _, _, v = torch.svd(cov)
            normals[i] = v[:, -1]

    return F.normalize(normals, dim=-1)
