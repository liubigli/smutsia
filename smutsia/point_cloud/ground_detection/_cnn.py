import os
import numpy as np
import torch
from matplotlib import pyplot as plt
from pyntcloud import PyntCloud
from smutsia.point_cloud.projection import Projection
from smutsia.point_cloud.normals import get_normals


def back_proj_front_pred(cloud, pred, proj, inference='regression'):
    """
    Parameters
    ----------
    cloud: Pyntcloud

    pred: np.ndarray

    proj: Projection

    inference: optional {'regression', 'classification'}
    """
    from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor

    # project gt
    acc_img = proj.project_points_values(cloud.xyz, np.ones(len(cloud.xyz)), aggregate_func='sum')

    acc_img_fl = acc_img.flatten()
    pred_fl = pred.flatten()

    lidx, i_img, j_img = proj.projector.project_point(cloud.xyz)
    # back project gt
    back_proj_acc_mask = acc_img_fl[lidx]
    back_proj_pred = pred_fl[lidx]

    num_neighbors = 3
    if inference == 'regression':
        knn = KNeighborsRegressor(n_neighbors=num_neighbors, weights='uniform')
    elif inference == 'classification':
        knn = KNeighborsClassifier(n_neighbors=num_neighbors)
    else:
        raise ValueError('Inference value can only be "regression" or "classification". '
                         'Passed value: {}'.format(inference))

    X_train, y_train = cloud.xyz[back_proj_acc_mask == 1, :], back_proj_pred[back_proj_acc_mask == 1]
    X_test = cloud.xyz[back_proj_acc_mask != 1, :]

    knn.fit(X_train, y_train)
    y_pred = knn.predict(X_test)
    labels = np.zeros_like(back_proj_pred)
    labels[back_proj_acc_mask == 1] = back_proj_pred[back_proj_acc_mask == 1]
    labels[back_proj_acc_mask != 1] = y_pred

    return labels


def project_cloud(cloud, proj, img_means, img_std, add_normals):
    """
    Parameters
    ----------
    cloud: PyntCloud

    proj: Projection

    img_means: np.ndarray

    img_std: np.ndarray

    add_normals: bool
    """
    # project point cloud on spherical image
    xyz = cloud.xyz
    i = cloud.points['i']
    z = xyz[:, 2]
    rho = np.linalg.norm(xyz, axis=1)
    aggr = ['max', 'min', 'mean']
    img_means = torch.from_numpy(img_means).clone()
    img_std = torch.from_numpy(img_std).clone()

    proj_img = proj.project_points_values(xyz, np.c_[rho, z, i], aggregate_func=aggr)

    # compute normals if necessary
    if add_normals:
        norm_img = get_normals(cloud=xyz, method='spherical', proj=proj, res_yaw=proj.res_yaw, res_pitch=proj.nb_layers)
        # stack information
        proj_img = np.dstack([proj_img, norm_img])
        # add to rescale values standard value for normals
        img_means = torch.cat([img_means, torch.zeros(3, dtype=img_means.dtype)])
        img_std = torch.cat([img_std, torch.ones(3, dtype=img_std.dtype)])

    # renormalize proj_img
    proj_img = torch.from_numpy(proj_img).clone()
    proj_img = proj_img.permute(2, 0, 1)
    proj_img = (proj_img - img_means[:, None, None]) / img_std[:, None, None]

    # return proj values
    return proj_img.unsqueeze(0)


def cnn_detect_ground(cloud, model, proj, img_means, img_std, add_normals, savedir='', gpu=0):
    """
    Parameters
    ----------
    cloud: np.ndarray or PyntCloud

    model: torch.nn.Module

    proj: Projection

    img_means: ndarray

    img_std: ndarray

    add_normals: bool

    savedir: str
        pass
    """
    # device = torch.device("cuda:"+str(gpu) if torch.cuda.is_available() else "cpu")
    device = torch.device("cpu")
    # project point cloud and extract normals
    proj_img = project_cloud(cloud=cloud, proj=proj, img_means=img_means, img_std=img_std, add_normals=add_normals)

    model.to(device)
    proj_img = proj_img.to(device=device, dtype=torch.float32)

    # evaluate point cloud
    with torch.no_grad():
        y_pred = model(proj_img)
        pred = torch.sigmoid(y_pred)
        thr_pred = (pred > 0.5).float()

    pred = np.asarray(pred[0, 0].to('cpu').detach())
    proj_img = np.asarray(proj_img[0].permute(1, 2, 0).to('cpu').detach())
    thr_pred = np.asarray(thr_pred[0, 0].to('cpu').detach())

    if len(savedir):
        fn = ''
        if hasattr(cloud, 'sequence'):
            fn += cloud.sequence + '_'
        if hasattr(cloud, 'filename'):
            fn += cloud.filename
        fn += '_2D'
        # todo save pred function
        fig, ax = plt.subplots(3, 1, figsize=(20, 4))
        if add_normals:
            ax[0].imshow(np.abs(proj_img[:, :, 3:]))
            ax[0].set_title('Estimated Normals')
        else:
            ax[0].imshow(np.abs(proj_img[:, :, 2]))
            ax[0].set_title('Reflectivity')
        ax[0].axis('off')
        ax[1].imshow(pred, cmap=plt.cm.coolwarm)
        ax[1].set_title('Heat Map')
        ax[1].axis('off')
        ax[2].imshow(thr_pred)
        ax[2].set_title('2D Prediction')
        ax[2].axis('off')
        fig.suptitle('Prediction on file {}'.format(fn))
        plt.tight_layout()
        plt.savefig(os.path.join(savedir, fn + '.eps'), dpi=90)

    # back project evaluated point cloud
    ground = back_proj_front_pred(cloud, pred, proj)

    return ground


if __name__ == "__main__":
    from glob import glob
    from smutsia.utils.semantickitti import load_pyntcloud
    from smutsia.utils.viz import plot_cloud
    from smutsia.deep_learning.models.u_net import UNet
    from definitions import SEMANTICKITTI_PATH

    weights = '/home/leonardo/Dev/github/smutsia/ckpt/ground_detection/unet_best.pth'
    net = UNet(n_channels=3, n_classes=1, n_filters=8, scale=(1, 2))
    net.load_state_dict(torch.load(weights))
    net.eval()

    layers_proj = Projection(proj_type='layers', nb_layers=64, res_yaw=2048)

    par_img_means = np.array([12.12, -1.04, 0.21])
    par_img_std = np.array([12.32, 0.86, 0.16])

    basedir = os.path.join(SEMANTICKITTI_PATH, '08', 'velodyne')

    files = sorted(glob(os.path.join(basedir, '*.bin')))

    pc = load_pyntcloud(files[0], add_label=True)

    out = cnn_detect_ground(pc, net, layers_proj, img_means=par_img_means, img_std=par_img_std, add_normals=False,
                            savedir='.')
    plot_cloud(pc.xyz, scalars=out, notebook=False)
    print("END")
