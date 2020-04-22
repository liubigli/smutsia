import numpy as np

def _logistic_function(x, x0, L, k):
    return L / (1 + np.exp(-k * (x - x0)))

def z_nz_dist(xyz, normals, src, dst):
    z = xyz[:, 2] - xyz[:, 2].min()
    nz = np.abs(normals[:,2])

    # apply logistic function to normals
    log_nz = _logistic_function(nz, x0=1/2, L=1, k=2)

    return np.abs((z[src] / log_nz[src]) - (z[dst] / log_nz[dst]))
