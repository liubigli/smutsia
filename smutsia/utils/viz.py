import itertools
import numpy as np
import pyvista as pv
import matplotlib as mpl
import colorsys
import matplotlib.colors as mc
from matplotlib import pyplot as plt
from matplotlib.collections import LineCollection
from mpl_toolkits.axes_grid1 import make_axes_locatable
from scipy.cluster.hierarchy import dendrogram, fcluster, set_link_color_palette
from sklearn.metrics.cluster import adjusted_rand_score as ri
from scipy.sparse import find

COLORS  = np.array(['#377eb8', '#ff7f00', '#4daf4a', '#a65628', '#f781bf', '#984ea3', '#999999', '#e41a1c', '#dede00'])


def plot_cloud(xyz,
               scalars=None,
               color=None,
               cmap=None,
               point_size=1.0,
               graph=None,
               rgb=False,
               add_scalarbar=True,
               interact=False,
               notebook=True,
               title=None,
               clim=None):
    """
    Helper functions

    Parameters
    ----------
    xyz: ndarray
        input point cloud

    scalars: ndarray
        array to use for coloring point cloud

    color:
        color to assign to all points

    cmap: matplotlib colormap
        matplotlib colormap to use

    point_size: float
        size of points in the plot

    graph: csr_matrix
        adjacent matrix representing a graph. The matrix columns and rows must the same of the number of points

    rgb: bool
        If True, it consider scalar values as RGB colors

    add_scalarbar: bool
        if True it adds a scalarbar in the plot

    interact: bool
        if true is possible to pick and select point during the plot

    notebook: bool
        set to True if plotting inside a jupyter notebook

    title: str
        plot title

    clim: list
        interval for color bar

    Returns
    -------
    plotter: pv.Plotter
        return plotter
    """
    if notebook:
        plotter = pv.BackgroundPlotter(title=title)
    else:
        plotter = pv.Plotter(title=title)

    poly = pv.PolyData(xyz)
    plotter.add_mesh(poly, color=color, scalars=scalars, cmap=cmap, rgb=rgb, point_size=point_size, clim=clim)

    if add_scalarbar and not rgb:
        plotter.add_scalar_bar()

    if graph is not None:
        src_g, dst_g, _ = find(graph)
        lines = np.zeros((2 * len(src_g), 3))
        for n, (s, d) in enumerate(zip(src_g, dst_g)):
            lines[2 * n, :] = xyz[s]
            lines[2 * n + 1, :] = xyz[d]
        actor = []

        def clear_lines(value):
            if not value:
                plotter.remove_actor(actor[-1])
                actor.pop(0)
            else:
                actor.append(plotter.add_lines(lines, width=1))
        plotter.add_checkbox_button_widget(clear_lines, value=False,
                                           position=(10.0, 10.0),
                                           size=10,
                                           border_size=2,
                                           color_on='blue',
                                           color_off='grey',
                                           background_color='white')

    def analyse_picked_points(picked_cells):
        # auxiliary function to analyse the picked cells
        ids = picked_cells.point_arrays['vtkOriginalPointIds']
        print("Selected Points: ")
        print(ids)
        print("Coordinates xyz: ")
        print(xyz[ids])

        if scalars is not None:
            print("Labels: ")
            print(scalars[ids])

    if interact:
        plotter.enable_cell_picking(through=False, callback=analyse_picked_points)

    if not notebook:
        plotter.show()

    return plotter


def color_bool_labeling(y_true, y_pred, pos_label=1, rgb=True):
    """
    Parameters
    ----------
    y_true: ndarray
        ground truth label

    y_pred: ndarray
        predicted labels

    pos_label: int
        Positive label value

    rgb: bool
        if True it returns rgb colors otherwise the points are coloured using grayscale color-bar.
    """
    if len(y_true) != len(y_pred):
        raise ValueError("y_true and y_pred don't have the same dimension.")

    base_shape = y_true.shape

    # true positives
    tp = np.logical_and(y_true == pos_label, y_pred == pos_label)
    # true negatives
    tn = np.logical_and(y_true == 0, y_pred == 0)
    # false positive
    fp = np.logical_and(y_true == 0, y_pred == pos_label)
    # false negatives
    fn = np.logical_and(y_true == pos_label, y_pred == 0)

    if rgb:
        colors = np.zeros(base_shape + (3,), dtype=np.uint8)
        colors[tp] = [23, 156, 82]  # green
        colors[tn] = [82, 65, 76]  # dark liver
        colors[fp] = [255, 62, 48]  # red
        colors[fn] = [23, 107, 239]  # blue
    else:
        colors = np.zeros_like(y_true, dtype=np.uint8)
        colors[tp] = 225
        colors[fn] = 150
        colors[fp] = 75

    return colors


def lighten_color(color_list, amount=0.25):
    """
    Lightens the given color by multiplying (1-luminosity) by the given amount.
    Input can be matplotlib color string, hex string, or RGB tuple.

    Examples:
    >> lighten_color('g', 0.3)
    >> lighten_color('#F034A3', 0.6)
    >> lighten_color((.3,.55,.1), 0.5)
    """
    out = []
    for color in color_list:
        try:
            c = mc.cnames[color]
        except:
            c = color
        c = colorsys.rgb_to_hls(*mc.to_rgb(c))
        lc = colorsys.hls_to_rgb(c[0], 1 - amount * (1 - c[1]), c[2])
        out.append(lc)
    return out


def plot_clustering(X, y, idx=None):
    ec = COLORS[y % len(COLORS)]
    plt.scatter(X[:, 0], X[:, 1], s=15, linewidths=1.5, c=lighten_color(ec), edgecolors=ec, alpha=0.9)
    # plt.axis([X[:,0].min(), X[:,0].max(), X[:,1].min(), X[:,1].max()])
    plt.xticks(())
    plt.yticks(())
    if idx is not None:
        iec = COLORS[y[idx] % len(COLORS)]
        plt.scatter(X[idx, 0], X[idx, 1], s=30, color=iec, marker='s', edgecolors='k')
    plt.xlim(X[:, 0].min(), X[:, 0].max())
    plt.ylim(X[:, 1].min(), X[:, 1].max())


def plot_dendrogram(linkage_matrix, n_clusters=0, lastp=30):
    extra = {} if lastp is None else dict(truncate_mode='lastp', p=lastp)
    set_link_color_palette(list(COLORS))
    dsort = np.sort(linkage_matrix[:, 2])
    dendrogram(linkage_matrix, no_labels=True, above_threshold_color="k", color_threshold=dsort[-n_clusters + 1],
               **extra)
    plt.yticks([])


def plot_graph(x, edge_index, edge_col):
    import torch
    if isinstance(x, torch.Tensor):
        xout = x.detach().cpu().numpy()
    else:
        xout = x

    if isinstance(edge_col, torch.Tensor):
        edge_col = edge_col.detach().cpu().numpy()
    else:
        edge_col = edge_col

    if isinstance(edge_index, torch.Tensor):
        e = edge_index.detach().cpu().numpy()
    else:
        e = edge_index

    segments = np.stack([xout[e[0]], xout[e[1]]], axis=1)
    lc = LineCollection(segments, zorder=0)
    lc.set_array(edge_col)
    lc.set_clim(vmin=0., vmax=1.0)
    ax = plt.gca()
    ax.set_xticks(())
    ax.set_yticks(())
    ax.set_xlim(segments[:, :, 0].min(), segments[:, :, 0].max())
    ax.set_ylim(segments[:, :, 1].min(), segments[:, :, 1].max())
    ax.add_collection(lc)
    axcb = plt.colorbar(lc)
    axcb.set_label('Edge Label')
    plt.sci(lc)
    plt.axis('equal')
    ax.scatter(xout[:, 0], xout[:, 1], s=20, c='w', edgecolors='k')


def plot_hyperbolic_eval(x, y, emb, linkage_matrix, emb_scale, y_pred=None, k=-1, show=True):
    """
    Auxiliary functions to plot results about hyperbolic clustering
    """
    n_clusters = y.max() + 1

    if k == -1:
        k = n_clusters

    if y_pred is None:
        y_pred = fcluster(linkage_matrix, n_clusters, criterion='maxclust') - 1

    k_ri_score = ri(y, y_pred)

    if k != n_clusters:
        y_pred_at_n = fcluster(linkage_matrix, n_clusters, criterion='maxclust') - 1
        val_ri_score = ri(y, y_pred_at_n)
        n_plots = 5
    else:
        val_ri_score = k_ri_score
        n_plots = 4
    # plot prediction
    idx = 1
    fig = plt.figure(figsize=(5 * n_plots, 5))
    ax = plt.subplot(1, n_plots, idx)
    plot_clustering(x, y)
    ax.set_title('Ground Truth')
    idx += 1
    ax = plt.subplot(1, n_plots, idx)
    plot_clustering(x, y_pred)
    ax.set_title(f'Pred: RI@{k}: {k_ri_score:.3f}')
    if k != n_clusters:
        idx += 1
        ax = plt.subplot(1, n_plots, idx)
        plot_clustering(x, y_pred_at_n)
        ax.set_title(f'Pred {n_clusters}: RI@{n_clusters}: {val_ri_score:.3f}')

    idx += 1
    ax = plt.subplot(1, n_plots, idx)
    plot_clustering(emb, y_pred)
    ax.set_xlim(-1 - 1e-1, 1 + 1e-1)
    ax.set_ylim(-1 - 1e-1, 1 + 1e-1)
    ax.set_title(f"Embeddings {emb_scale}")
    idx += 1
    ax = plt.subplot(1, n_plots, idx)
    plot_dendrogram(linkage_matrix, n_clusters=k)
    ax.set_title(f'Dendrogram {k}-clusters')
    plt.tight_layout()
    if show:
        plt.show()
    else:
        return fig

def inspect_missclass(im_class, im_pred, selected_id, im_max, im_min, savedir, filename):
    """128 ground and bike! other bikes with their label.  The rest 0. We
    can evaluate which bike pixels are invaded by the ground.

    In fact, it works for any class (not only bikes). selectedId is a parameter
    """
    import smilPython as sm
    import os
    my_lut = sm.Map_UINT8_UINT8()

    for i in range(256):
        my_lut[i] = 0

    imtmp = sm.Image(im_class)

    for elem in selected_id:
        my_lut[elem] = elem

    # imtmp has sel label on selected_id or 0 elsewhere
    sm.applyLookup(im_class, my_lut, imtmp)

    if sm.maxVal(imtmp) > 0:
        # if the image contains the class of interest
        imtmp2 = sm.Image(imtmp)
        sm.compare(im_pred, ">", 0, 128, 0, imtmp2)
        # only select_id pixels
        sm.compare(imtmp, "==", 0, 0, imtmp2, imtmp2)
        sm.compare(imtmp2, ">", 0, imtmp2, imtmp, imtmp2)
        # 128 pred class and selected id!
        sm.write(imtmp2, os.path.join(savedir, filename + "_res.png"))
        sm.dilate(imtmp, imtmp, sm.HexSE(2))
        sm.compare(imtmp, ">", 0, im_max, 0, imtmp2)
        sm.write(imtmp2, os.path.join(savedir, filename + "_max.png"))
        sm.compare(imtmp, ">", 0, im_min, 0, imtmp2)
        sm.write(imtmp2, os.path.join(savedir, filename + "_min.png"))
        return 1
    else:
        return 0


def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues,
                          figsize=(8, 8),
                          savefig=""):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.

    Parameters
    ----------
    cm: ndarray
        input confusion matrix

    classes: list
        list of classes to use as xticks and yticks in the plot

    normalize: bool
        if true normalise confusion before plotting it

    title: str
        title in the plot

    cmap: plt.cm
        colormap to use

    figsize: tuple
        size of the figure

    savefig: str
        filename to give if you want to save the figure

    """
    plt.style.use('ggplot')
    font = {
        'family': 'arial',
        'size': 14}

    mpl.rc('font', **font)
    if normalize:
        from smutsia.utils.scores import mat_renorm_rows
        cm_plot = mat_renorm_rows(cm)
    else:
        cm_plot = cm

    plt.figure(figsize=figsize)
    ax = plt.gca()
    im = ax.imshow(cm_plot, interpolation='nearest', cmap=cmap)
    ax.grid(False)

    plt.title(title)

    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = 0.5 if normalize else cm_plot.max() / 2
    for i, j in itertools.product(range(cm_plot.shape[0]), range(cm_plot.shape[1])):
        plt.text(j, i, format(cm_plot[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm_plot[i, j] > thresh else "black")
    plt.ylabel('True label')
    plt.xlabel('Predicted label')

    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.5)
    plt.colorbar(im, cax=cax)
    plt.tight_layout()
    if len(savefig) > 0:
        plt.savefig(savefig, dpi=90)

    # restore style to default settings
    mpl.rcParams.update(mpl.rcParamsDefault)


def plot_precision_recall_curve(prec_scores, recall_scores, figsize=(12, 12),
                                xlim=None, ylim=None, title='', savefig='', filename=''):
    if len(title) == 0:
        title = 'Precision-Recall curve ' + filename

    if xlim is None:
        xlim = [0.75, 1.0]

    if ylim is None:
        ylim = [0.75, 1.0]
    # In matplotlib < 1.5, plt.fill_between does not have a 'step' argument

    plt.figure(figsize=figsize)
    plt.style.use('ggplot')
    plt.step(prec_scores, recall_scores, linewidth=2, color='tab:blue', where='post')
    plt.xlabel('Recall', fontsize=22)
    plt.ylabel('Precision', fontsize=22)
    plt.ylim(ylim)
    plt.xlim(xlim)
    plt.title(title, fontsize=24)
    # plt.legend(legend_list, loc=3, prop={'size': 18})
    plt.tight_layout()
    if len(savefig):
        plt.savefig(savefig, dpi=90)
