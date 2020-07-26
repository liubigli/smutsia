import torch
import numpy as np
from scipy.spatial import cKDTree
from torch_geometric.nn import knn_graph

def rri_representations(xyz, k=10):
    """
    Compute knn representation of input point cloud
    
    Parameters
    ----------
    xyz: np.ndarray
        input points
        
    k: int
        number of nn to keep into account to extract knn_representations
    """
    n_points = len(xyz)
    rho = np.linalg.norm(xyz, axis=1)
    xyz_n = np.divide(xyz, rho.reshape(-1, 1))
    tree = cKDTree(xyz)
    dist, neighs = tree.query(xyz, k=k+1)
    knn = neighs[:, 1:]

    # computing thetas
    # X shape is Nk x 3
    X = xyz_n.repeat(k, axis=0)
    Y = xyz_n[knn.flatten()]  # Y shape is Nk x 3
    # element wise product and then sum along axis 1
    thetas = np.einsum('ij,ij->i', X, Y)  # theta shape is Nk
    thetas = np.arccos(thetas)
    thetas = thetas.reshape(-1, k)  # reshape to obtain N x k matrix

    # compute phis
    Y1 = xyz[knn.flatten()]  # Nk x 3
    # element wise product and then sum along axis 1
    XY1 = np.einsum('ij,ij->i', X, Y1)  # XY1 shape is Nk
    # print(np.c_[XY1, XY1, XY1] * X)
    Tp = (Y1 - np.c_[XY1, XY1, XY1] * X)  # Tp shape is Nk x 3
    Tp_n = np.divide(Tp, np.linalg.norm(Tp, axis=1).reshape(-1, 1)).reshape((n_points, k, 3))  # Tp shape is Nk x 3

    cos_phi = np.einsum('ijh,ikh->ijk', Tp_n, Tp_n)

    # # todo: I don't know how to compute this faster
    sin_phi = np.sqrt(np.clip(1 - cos_phi**2, a_min=0, a_max=1))
    # sin_phi = np.zeros_like(cos_phi)
    # for i in range(n_points):
    #     for j in range(k):
    #         for h in range(k):
    #             sin_phi[i, j, h] = np.cross(Tp_n[i, j], Tp_n[i, h]).dot(xyz_n[i])

    cos_phi = cos_phi.flatten()  # NxKxK
    sin_phi = sin_phi.flatten()  # NxKxK
    phi = np.arctan2(sin_phi, cos_phi)
    phi = phi.reshape(n_points, k, k)  # NxKxK
    phis = np.zeros((n_points, k))

    k_ids = np.arange(k)
    # todo improve this too
    for h in range(k):
        phis[:, h] = phi[:, h, k_ids != h].min(axis=1)

    rri_repr = np.zeros((n_points, k, 4), dtype=np.float)
    rri_repr[:, :, 0] = rho.reshape(-1, 1).repeat(k, axis=1)
    rri_repr[:, :, 1] = rho[knn]
    rri_repr[:, :, 2] = thetas
    rri_repr[:, :, 3] = phis

    return rri_repr

#
def batch_knn_graph(x, k, batch=None, loop=False, flow='source_to_target', cosine=False):
    if batch is not None:
        unique, counts = torch.unique(batch, return_counts=True)
        edge_index = torch.cat([knn_graph(x[batch==u], k=k, loop=loop, flow=flow, cosine=cosine) + u * c for u, c in zip(unique, counts)], dim=1)
        return edge_index
    else:
        return knn_graph(x, k, loop=loop, flow=flow, cosine=cosine)

def torch_rri_representations(xyz, k, batch=None):
    """
    Pythorch implementation
    Compute knn representation of input point cloud

    Parameters
    ----------
    xyz: torch.Tensor
        input points

    k: int
        number of nn to keep into account to extract knn_representations
    """
    eps = 1e-15
    n_points = xyz.size(0)
    rho = xyz.norm(dim=1)
    xyz_n = torch.div(xyz, rho.reshape(-1, 1))
    edge_index = batch_knn_graph(xyz, k=k, loop=False, batch=batch)
    dst, src = edge_index[0], edge_index[1]
    knn = dst
    # computing thetas
    # X shape is Nk x 3
    X = torch.repeat_interleave(xyz_n, k, dim=0)
    Y = xyz_n[knn.flatten()]  # Y shape is Nk x 3
    # element wise product and then sum along axis 1
    thetas = torch.einsum('ij,ij->i', X, Y)  # theta shape is Nk
    thetas = torch.acos(torch.clamp(thetas, min=-1.0, max=1.0))
    thetas = thetas.reshape(-1, k)  # reshape to obtain N x k matrix

    # compute phis
    Y1 = xyz[knn.flatten()]  # Nk x 3
    # element wise product and then sum along axis 1
    XY1 = torch.einsum('ij,ij->i', X, Y1)  # XY1 shape is Nk
    # print(np.c_[XY1, XY1, XY1] * X)
    Tp = (Y1 - XY1.reshape(-1, 1) * X)  # Tp shape is Nk x 3
    Tp_n = torch.div(Tp, Tp.norm(dim=1).reshape(-1, 1)).reshape((n_points, k, 3))  # Tp shape is Nk x 3

    cos_phi = torch.einsum('ijh,ikh->ijk', Tp_n, Tp_n)

    # # todo: I don't know how to compute this faster
    sin_phi = torch.sqrt(torch.clamp(1 - cos_phi ** 2, min=eps, max=1))
    # sin_phi = cos_phi

    cos_phi = cos_phi.flatten()  # NxKxK
    sin_phi = sin_phi.flatten()  # NxKxK
    phi = torch.atan2(sin_phi, cos_phi)
    phi = phi.reshape(n_points, k, k)  # NxKxK
    phis = torch.zeros((n_points, k))

    k_ids = torch.arange(k)
    # todo improve this too
    for h in range(k):
        phis[:, h] = phi[:, h, k_ids != h].min(dim=1).values

    rri_repr = torch.zeros((n_points, k, 4), dtype=torch.float)
    rri_repr[:, :, 0] = rho.reshape(-1, 1).repeat_interleave(k, dim=1)
    rri_repr[:, :, 1] = rho[knn].reshape(n_points, k)
    rri_repr[:, :, 2] = thetas
    rri_repr[:, :, 3] = phis

    return rri_repr, edge_index

## test
if __name__ == "__main__":
    import time
    from tqdm import tqdm
    from definitions import MODEL_NET
    from torch_geometric.data import DataLoader
    from torch_geometric.datasets import ModelNet
    from torch_geometric.transforms import SamplePoints, NormalizeScale

    modelnet= ModelNet(root=MODEL_NET, name='40', transform=SamplePoints(1024), pre_transform=NormalizeScale())
    for i in range(0, 10):
        print(f"Batch {i*1000}->{(i+1) * 1000}")
        loader = DataLoader(modelnet[i * 1000: (i+1) * 1000] , batch_size=8, shuffle=False)
        with tqdm(total=len(loader), desc='ModelNet 40') as pbar:
            for it, data in enumerate(loader):
                pos = data.pos
                batch = data.batch
                rri, edge_index = torch_rri_representations(pos, k=20, batch=None)
                nan_ax0 = torch.isnan(rri[:, :, 0]).sum().item()
                nan_ax1 = torch.isnan(rri[:, :, 1]).sum().item()
                nan_ax2 = torch.isnan(rri[:, :, 2]).sum().item()
                nan_ax3 = torch.isnan(rri[:, :, 3]).sum().item()
                pbar.set_postfix(**{
                    'nan axis: 0': nan_ax0,
                    'nan axis: 1': nan_ax1,
                    'nan axis: 2': nan_ax2,
                    'nan axis: 3': nan_ax3,
                })
                assert torch.isnan(rri[:, :, 0]).sum().item() == 0
                assert torch.isnan(rri[:, :, 1]).sum().item() == 0
                assert torch.isnan(rri[:, :, 2]).sum().item() == 0
                assert torch.isnan(rri[:, :, 3]).sum().item() == 0
                pbar.update()
    # for i in range(10):
    #     x = torch.rand(100, 3)
    #     start = time.time()
    #     pt_rri = torch_rri_representations(x, k=10)
    #     stop1 = time.time()
    #     np_rry = rri_representations(x.detach().numpy(), k=10)
    #     stop2 = time.time()
    #     print(np.abs(pt_rri.numpy()-np_rry).sum(), f"Torch Time {stop1-start}. Np Time {stop2 - stop1}")
