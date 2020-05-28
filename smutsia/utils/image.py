from __future__ import print_function
from __future__ import division
from __future__ import absolute_import

import time
import numpy as np
from numpy import matlib
import smilPython as sm
from scipy.sparse import csr_matrix, find
from matplotlib import pyplot as plt
from sklearn.feature_extraction.image import _make_edges_3d


def pixel_to_node(p, ishape, order='C'):
    """
    From pixel coordinates to associated node in the image graph

    Parameters
    ----------
    p: tuple
        Coordinates (i,j) of the pixel

    ishape: tuple
        Size of the image

    order : {'C', 'F', 'A'}, optional
        Read the elements of `a` using this index order, and place the
        elements into the reshaped array using this index order.  'C'
        means to read / write the elements using C-like index order,
        with the last axis index changing fastest, back to the first
        axis index changing slowest. 'F' means to read / write the
        elements using Fortran-like index order, with the first index
        changing fastest, and the last index changing slowest. Note that
        the 'C' and 'F' options take no account of the memory layout of
        the underlying array, and only refer to the order of indexing.
        'A' means to read / write the elements in Fortran-like index
        order if `a` is Fortran *contiguous* in memory, C-like order
        otherwise.

    Returns
    -------
    node: int
        Id of the corresponding node in the graph associated to image

    """

    if order == 'C':  # C-like index order
        return p[0] * ishape[1] + p[1]

    if order == 'F':  # Fortran-like index order
        return p[0] + p[1] * ishape[0]


def node_to_pixel(n, ishape, order='C'):
    """
    From node in image graph to associated pixel

    Parameters
    ----------
    n: int
        Id of the corresponding node in the graph associated to image

    ishape: tuple
        Size of the image

    order : {'C', 'F', 'A'}, optional
        Read the elements of `a` using this index order, and place the
        elements into the reshaped array using this index order.  'C'
        means to read / write the elements using C-like index order,
        with the last axis index changing fastest, back to the first
        axis index changing slowest. 'F' means to read / write the
        elements using Fortran-like index order, with the first index
        changing fastest, and the last index changing slowest. Note that
        the 'C' and 'F' options take no account of the memory layout of
        the underlying array, and only refer to the order of indexing.
        'A' means to read / write the elements in Fortran-like index
        order if `a` is Fortran *contiguous* in memory, C-like order
        otherwise.

    Returns
    -------
    node: tuple
        Coordinates of the corresponding pixel in the image

    """

    i, j = 0, 0  # initializing returning variables

    if order == 'C':  # C-like index order
        i = np.floor(n / ishape[1]).astype(int)
        j = n % ishape[1]

    if order == 'F':  # Fortran-like index order
        i = n % ishape[0]
        j = np.floor(n / ishape[0]).astype(int)

    return i, j


def _img_to_4_connected_graph(img):
    """
    Function that returns the 4 connected weighted graph associated to the input image.
    The weights of the edges are the differences between the pixels.
    This is a simpler version of img_to_graph function implemented below

    Parameters
    ----------
    img: ndarray
        input image

    Returns
    -------
    graph: csr_matrix
        graph associated to image

    """
    # implemented by Santiago-Velasco-Forero
    # convert input image to 3d image
    img = np.atleast_3d(img)
    # get image shape
    nr, nc, nz = img.shape
    # defining little eps in order to have also zero values edges
    eps = 1e-10
    # get edges for
    edges = _make_edges_3d(nr, nc, n_z=1)

    if nz == 1:
        imgtemp = img.flatten().astype('float')
        # consider 4 connectivity
        grad = abs(imgtemp[edges[0]] - imgtemp[edges[1]]) + eps
    else:
        # in case of coloured images we use the maximum of color differences
        # among all channels as the edge weights
        # we copy images
        imgtemp = img.reshape(-1, nz).astype('float')
        grad = np.abs(imgtemp[edges[0]] - imgtemp[edges[1]]) + 1e-10
        grad = grad.max(axis=1)

    return csr_matrix((grad, (edges[0], edges[1])), shape=(nr * nc, nr * nc))


def _make_edges_spherical_images(nr, nc):
    """
    Function that generate a graph for spherical images

    Parameters
    ----------
    nr: int
        number of rows
    nc: int
        number of columns

    Returns
    -------
    edges: ndarray
    """
    # number of row and number of columns in the image
    edges = _make_edges_3d(nr, nc, n_z=1)
    # we have to add edges on between the pixels on the left side and the ones in the rights sides
    edges_to_add = (np.vstack([np.arange(0, nr * nc, nc), np.arange(nc - 1, nr * nc, nc)]))
    return np.append(edges, edges_to_add, axis=1)


def img_to_graph(img, metric=None, order='C', **kwargs):
    """
    Function that return a 4-connected graph from an image on which the weights are defined according to a given metric

    Parameters
    ----------
    img: (N,M) ndarray
        input image

    metric: function (default None)
        metric used to assign weights to edges

    order : {'C', 'F', 'A', 'K'}, optional
        'C' means to flatten in row-major (C-style) order.
        'F' means to flatten in column-major (Fortran-style) order.
        'A' means to flatten in column-major order if `a` is Fortran *contiguous* in memory, row-major order otherwise.
        'K' means to flatten `a` in the order the elements occur in memory.
        The default is 'C'.

    kwargs: dict
        supplemental arguments for the metric function

    Returns
    -------
    graph: (N*M, N*M) csr_matrix
        adjacency matrix of the graph associated to image img
    """
    # implemented by Santiago-Velasco-Forero

    img = np.atleast_3d(img)

    nr, nc, nz = img.shape

    # default case: we build a 4-connected graph
    if order == 'C':  # order for image pixels K=columns-major order/ C=row-major order
        edges = _make_edges_3d(nr, nc, n_z=1)
    else:
        edges = _make_edges_3d(nc, nr, n_z=1)

    # we copy images
    imgtemp = img.copy().astype('float')
    # we flatten only first two dimension and imgtemp is a {nr*nc} X {nz} vector
    if nz == 1:
        imgtemp = imgtemp.flatten(order=order)
    else:
        imgtemp = imgtemp.reshape(-1, nz, order=order)

    if metric is None:
        # default metric is the gradient metric
        weights = np.abs(imgtemp[edges[0]] - imgtemp[edges[1]]) + 1e-10
        if nz > 1:
            # in case of color/multispectral images we take as distance the max distance in all channels
            weights = weights.max(axis=1)
        else:
            # gray level image
            weights = weights.flatten()
    else:
        # otherwise is possible to pass a custom function to compute distances between two adjacent pixels
        weights = np.array([metric(imgtemp, e[0], e[1], **kwargs) for e in edges.T])

    # to save space we delete temp image
    del imgtemp

    # is possible to define a custom mask and remap edges according to this map
    mask = kwargs.get('mask', None)

    if mask is not None:
        # graph shape is determined by the max value of mask
        g_shape = (mask.max() + 1, mask.max() + 1)

        # in case of custom mask we return the graph
        return csr_matrix((weights, (mask[edges[0]], mask[edges[1]])), shape=g_shape)

    # otherwise is possible to define an offset on the graph in order to manage the case of split images
    # the offset is defined according to the order of flattening of the image.
    # So is always defined in a unique direction and it allows us to define a graph of the good dimension
    g_col = nr * nc

    offset = 0
    # switch in case of row-major or column-major order
    if order == 'C':
        # case in which the matrix is flatten in c-style i.e. row-major order
        row_offset = kwargs.get('row_offset', 0)
        offset = row_offset * nc
        # updating graph dimension
        g_col = (nr + row_offset) * nc

    elif order == 'F':
        # case in which the matrix is flatten in fortran-style i.e. columns-major order
        col_offset = kwargs.get('col_offset', 0)
        offset = col_offset * nr
        g_col = nr * (nc + col_offset)

    # graph associated to images are always represented as square sparse matrices
    g_shape = (g_col, g_col)

    # we return the computed graph as csr_matrix
    return csr_matrix((weights, (offset + edges[0], offset + edges[1])), shape=g_shape)


def plot_graph(img,
               graphs,
               labels=None,
               filename="",
               figsize=(8, 8),
               order='C',
               saveplot=False,
               colors=None):
    """
    Function that plots up to 5 graphs contained in the list graphs

    Parameters
    ----------
    img: ndarray
        Image to plot

    graphs: list
        Graphs to plot

    labels: list
        list of labels that remaps nodes in list of graphs

    filename: string
        Name of the file to save

    figsize: tuple
        Dimension of the figure to save

    order : {'C', 'F', 'A', 'K'}, optional
        'C' means to flatten in row-major (C-style) order.
        'F' means to flatten in column-major (Fortran-style) order.
        'A' means to flatten in column-major order if `a` is Fortran *contiguous* in memory, row-major order otherwise.
        'K' means to flatten `a` in the order the elements occur in memory.
        The default is 'C'

    saveplot: bool
        Set to True to save the file. Default is False.

    colors: list
        list of colors for graphs. Default ['g', 'r', 'b', 'k', 'm'].
    """
    # thanks to an idea of Santiago-Velasco-Forero
    if type(graphs) is not list:
        graphs = [graphs]

    if labels is not None and type(labels) is not list:
        labels = [labels]

    if colors is None:
        colors = ['g', 'r', 'b', 'k', 'm']

    nr, nc = img.shape[:2]
    n_plots = min(len(graphs), len(colors))
    edges_list = [[]] * n_plots
    for i in range(n_plots):
        edges_list[i] = find(graphs[i])
    if order == 'F':
        dx = np.matlib.repmat(np.arange(nc), nr, 1).transpose()
        dy = np.matlib.repmat(np.arange(nr), nc, 1)
    else:
        dx = np.matlib.repmat(np.arange(nr), nc, 1).transpose()
        dy = np.matlib.repmat(np.arange(nc), nr, 1)
    dx = dx.flatten()
    dy = dy.flatten()
    plt.figure(figsize=figsize)
    plt.gcf()
    plt.gca()

    plt.imshow(img)
    plt.tight_layout(pad=0)
    plt.axis('off')

    for i in range(n_plots):
        if labels is not None:
            imi, jmi = labels[i][edges_list[i][0]], labels[i][edges_list[i][1]]
        else:
            imi, jmi = edges_list[i][0], edges_list[i][1]
        if order == 'F':
            plt.plot([dx[imi], dx[jmi]], [dy[imi], dy[jmi]], '-' + colors[i])
        else:
            plt.plot([dy[imi], dy[jmi]], [dx[imi], dx[jmi]], '-' + colors[i])

    if saveplot:
        plt.savefig(filename + '_' + time.strftime('%s') + ".png", bbox_inches='tight', pad_inches=0, transparent=True)
        plt.close()
    else:
        plt.show()

    plt.show()


def plot_sub_graph(img, graph, min_row, max_row, min_col, max_col, figsize=(8, 8), order='C', colors=None):
    """
    Function that plot a subgraph of the main graph

    Parameters
    ----------
    img: ndarray
        input image

    graph: csr_matrix
        input graph

    min_col: int
        minimum value of x interval

    max_col: int
        max value for the x interval

    min_row: int
        min value for the y interval

    max_row: int
        max value for the y interval

    figsize: tuple
        Size of the figure

    colors: list
        colors to use for subgraph

    order : {'C', 'F', 'A', 'K'}, optional
        'C' means to flatten in row-major (C-style) order.
        'F' means to flatten in column-major (Fortran-style) order.
        'A' means to flatten in column-major order if `a` is Fortran *contiguous* in memory, row-major order otherwise.
        'K' means to flatten `a` in the order the elements occur in memory.
        The default is 'C'.
    """
    # the basic idea of this function is to select pixels contained in the interval [ymin:ymax, xmin:xmax]
    # and the corresponding nodes and edges in the graphs. Once selected we call the function plot_graph
    nr, nc = img.shape[:2]  # getting image shapes

    # selecting subimg
    subimg = img[min_row:max_row, min_col:max_col]

    # getting subimage dimensions
    # snr, snc = subimg.shape[:2]

    # selecting nodes corresponding to the pixels in the graph
    graph_nodes = np.arange(nr * nc).reshape(nr, nc, order=order)[min_row:max_row, min_col:max_col].flatten(order=order)
    # # cartesian product of the nodes ids
    # cart = np.array(np.meshgrid(graph_nodes, graph_nodes)).T.reshape(-1, 2)

    # selecting subgraph
    # sub = csr_matrix(graph[cart[:, 0], cart[:, 1]].reshape(snr * snc, snr * snc))
    if type(graph) is list:
        sub = []
        for g in graph:
            g_rows, g_cols = g.shape
            sub.append(g[graph_nodes[graph_nodes < g_rows], :][:, graph_nodes[graph_nodes < g_cols]])
    else:
        g_rows, g_cols = graph.shape
        sub = graph[graph_nodes[graph_nodes < g_rows], :][:, graph_nodes[graph_nodes < g_cols]]
    # plotting subimage and subgraph
    plot_graph(subimg, sub, figsize=figsize, order=order, colors=colors)


def accumarray(indices, vals, size, func='plus', fill_value=0):
    """
    from: https://github.com/pathak22/videoseg/blob/master/src/utils.py
    Implementing python equivalent of matlab accumarray.
    Taken from SDS repo: master/superpixel_representation.py#L36-L46

    Parameters
    ----------
    indices: ndarray
        must be a numpy array (any shape)

    vals: ndarray
        numpy array of same shape as indices or a scalar

    size: int
        must be the number of diffent values

    func: {'plus', 'minus', 'times', 'max', 'min', 'and', 'or'} optional
        Default is 'plus'

    fill_value: int
        Default is 0

    """

    # get dictionary
    function_name_dict = {
        'plus': (np.add, 0.),
        'minus': (np.subtract, 0.),
        'times': (np.multiply, 1.),
        'max': (np.maximum, -np.inf),
        'min': (np.minimum, np.inf),
        'and': (np.logical_and, True),
        'or': (np.logical_or, False)}

    if func not in function_name_dict:
        raise KeyError('Function name not defined for accumarray')

    if np.isscalar(vals):
        if isinstance(indices, tuple):
            shape = indices[0].shape
        else:
            shape = indices.shape
        vals = np.tile(vals, shape)

    # get the function and the default value
    (func, value) = function_name_dict[func]

    # create an array to hold things
    output = np.ndarray(size)
    output[:] = value
    func.at(output, indices, vals)

    # also check whether indices have been used or not
    isthere = np.ndarray(size, 'bool')
    istherevals = np.ones(vals.shape, 'bool')
    (func, value) = function_name_dict['or']
    isthere[:] = value
    func.at(isthere, indices, istherevals)

    # fill things that were not used with fill value
    output[np.invert(isthere)] = fill_value

    return output


def label_image(img, labels, order='C'):
    """
    Function that given an image and a vector of labels for its pixels returns the corresponding segmented image

    Parameters
    ----------
    img: nxm ndarray
        input image
    labels: n*m ndarray
        Array of labels for pixels of the input image.

    order : {'C', 'F', 'A', 'K'}, optional
        'C' means to flatten in row-major (C-style) order.
        'F' means to flatten in column-major (Fortran-style) order.
        'A' means to flatten in column-major order if `a` is Fortran *contiguous* in memory, row-major order otherwise.
        'K' means to flatten `a` in the order the elements occur in memory.
        The default is 'C'.

    Returns
    -------
    img_label: nxm ndarray
        Resulting segmented image
    """
    img = np.atleast_3d(img)
    # image dimensions
    nr, nc, nz = img.shape

    n_cc = labels.max() + 1

    s = []
    for i in range(nz):
        s.append(accumarray(labels, img[:, :, i].flatten(order=order), n_cc, func='plus'))

    ne = accumarray(labels, np.ones(nr*nc), n_cc, func='plus')

    for i in range(nz):
        s[i] = s[i] / ne
        s[i] = (s[i][labels]).reshape((nr, nc), order=order)

    img_label = np.zeros(img.shape)

    for i in range(nz):
        img_label[:, :, i] = s[i]

    if nz == 1:
        return img_label[:, :, 0]
    else:
        return img_label


def stick_two_images(img1, img2, num_overlapping=0, direction='H'):
    """
    Function that sticks two different images

    img1: ndarray
        First image to stick

    img2: ndarray
        Second image to stick

    num_overlapping: int
        number of overlapping rows or columns

    direction: {'H', 'V'} optional
        Stick direction.
        'H' means horizontal direction of sticking, i.e. the images are one near the other
        'V' means vertical direction of sticking, i.e. the images are one above the other

    Returns
    -------
    merged_img: MxN ndarray

    """
    img1 = np.atleast_3d(img1)
    img2 = np.atleast_3d(img2)

    # getting shape of the two images
    nr1, nc1, nz1 = img1.shape
    nr2, nc2, nz2 = img2.shape

    # if nr1*nc1*nz1*nr2*nc2*nz2 == 0:
    #     raise ValueError('negative dimensions are not allowed')

    if direction.lower() == 'h':
        if nr1 != nr2 or nz1 != nz2:
            raise ValueError('dimension mismatch: the two images have a different number of rows or channels')

        merged_img = np.zeros((nr1, nc1 + nc2 - num_overlapping, nz1), dtype=img1.dtype)
        merged_img[:, :nc1] = img1
        merged_img[:, nc1 - num_overlapping:] = img2

        if nz1 > 1:
            return merged_img
        else:
            return merged_img[:, :, 0]

    if direction.lower() == 'v':
        if nc1 != nc2 or nz1 != nz2:
            raise ValueError('Dimension mismatch! The two images have a different number of rows or channels')

        merged_img = np.zeros((nr1 + nr2 - num_overlapping, nc1, nz1), dtype=img1.dtype)
        merged_img[:nr1, :] = img1
        merged_img[nr1 - num_overlapping:, :] = img2

        if nz1 > 1:
            return merged_img
        else:
            return merged_img[:, :, 0]

    else:
        raise ValueError('Direction of merging not known')


def smil_2_np(im):
    """
    Auxiliary function to convert a smil image 2 numpy image

    Parameters
    ----------
    im: sm.Image
        input smil image

    Returns
    -------
    im_array: ndarray
        numpy image
    """
    # get image content
    im_array = im.getNumArray()
    # swap rows with columns
    im_array = np.swapaxes(im_array, 0, 1)
    return im_array


def np_2_smil(np_img):
    """
    Auxiliary function to convert a numpy img to smil image

    Parameters
    ----------
    np_img: ndarray
        input image

    Returns
    -------
    smil_im: sm.Image
        smil image
    """
    np_swap_img = np.swapaxes(np_img, 0, 1)

    im_shape = np_swap_img.shape

    if np_img.dtype == 'uint8':
        smil_im = sm.Image(im_shape[0], im_shape[1])
    elif np_img.dtype == 'uint16':
        temp_img = sm.Image(im_shape[0], im_shape[1])
        smil_im = sm.Image(temp_img, 'UINT16')
    else:
        Warning("Warning: {} copied to uint8".format(np_img.dtype))
        smil_im = sm.Image(im_shape[0], im_shape[1])

    imArray = smil_im.getNumArray()

    imArray[:, :] = np_swap_img

    return smil_im


def label_with_measure(im, im_val, im_out, measure_str, nl=sm.Morpho.getDefaultSE()):
    # ----------------------------------------
    # Compute Blobs
    # ----------------------------------------
    im_label = sm.Image(im, "UINT16")
    sm.label(im, im_label, nl)
    blobs = sm.computeBlobs(im_label)

    if measure_str == "mean":
        meas_list = sm.measMeanVals(im_val, blobs)
    elif measure_str == "max":
        meas_list = sm.measMaxVals(im_val, blobs)
    elif measure_str == "min":
        meas_list = sm.measMinVals(im_val, blobs)
    elif measure_str == "mode":
        meas_list = sm.measModeVals(im_val, blobs)
    elif measure_str == "median":
        meas_list = sm.measMedianVals(im_val, blobs)
    elif measure_str == "nb":
        meas_list = sm.measNbVals(im_val, blobs)
    elif measure_str == "nbLab":
        meas_list = sm.measNbLabVals(im_val, blobs)
    else:
        raise ValueError("measure_str value {} not valid".format(measure_str))

    my_lut = sm.Map_UINT16_UINT16()
    if measure_str == "mean":
        for lbl in blobs.keys():
            my_lut[lbl] = int(meas_list[lbl][0])
    else:  # min,max...
        for lbl in blobs.keys():
            my_lut[lbl] = int(meas_list[lbl])
    imtmp16 = sm.Image(im_label)

    sm.applyLookup(im_label, my_lut, imtmp16)
    sm.copy(imtmp16, im_out)

