import itertools
import numpy as np
import pyvista as pv
import matplotlib as mpl
from matplotlib import pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
from scipy.sparse import find


def plot_cloud(xyz,
               scalars=None,
               color=None,
               cmap=None,
               point_size=1.0,
               graph=None,
               rgb=False,
               interact=False,
               notebook=True):
    """
    Helper functions
    Parameters
    ----------
    """
    if notebook:
        plotter = pv.BackgroundPlotter()
    else:
        plotter = pv.Plotter()

    poly = pv.PolyData(xyz)
    plotter.add_mesh(poly, color=color, scalars=scalars, cmap=cmap, rgb=rgb, point_size=point_size)
    plotter.add_scalar_bar()
    if graph is not None:
        src_g, dst_g, _ = find(graph)
        lines = np.zeros((2 * len(src_g), 3))
        for n, (s, d) in enumerate(zip(src_g, dst_g)):
            lines[2 * n, :] = xyz[s]
            lines[2 * n + 1, :] = xyz[d]
        actor = []
        # actor.append(plotter.add_lines(lines, width=1))
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


    # auxiliary function to analyse the picked cells
    def analyse_picked_points(picked_cells):
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


def color_bool_labeling(y_true, y_pred, pos_label=1):
    """
    Parameters
    ----------
    y_true: ndarray
        ground truth label

    y_pred: ndarray
        predicted labels

    pos_label: int
        Positive label value
    """
    if len(y_true) != len(y_pred):
        raise ValueError("y_true and y_pred don't have the same dimension.")

    base_shape = y_true.shape
    colors = np.zeros(base_shape + (3, ), dtype=np.uint8)
    # true positives
    tp = np.logical_and(y_true == pos_label, y_pred == pos_label)
    # true negatives
    tn = np.logical_and(y_true == 0, y_pred == 0)
    # false positive
    fp = np.logical_and(y_true == 0, y_pred == pos_label)
    # false negatives
    fn = np.logical_and(y_true == pos_label, y_pred == 0)

    colors[tp] = [23, 156, 82]  # green
    colors[tn] = [82, 65, 76]  # dark liver
    colors[fp] = [255, 62, 48]  # red
    colors[fn] = [23, 107, 239]  # blue

    return colors


def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues,
                          figsize=(8, 8),
                          savefig=""):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
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
