import numpy as np
import pyvista as pv
from scipy.sparse import find

def plot_cloud(xyz, scalars=None, cmap=None, point_size=1.0, graph=None, interact=False):
    """
    Helper functions
    Parameters
    ----------
    """
    plotter = pv.BackgroundPlotter()
    poly = pv.PolyData(xyz)
    plotter.add_mesh(poly, scalars=scalars, cmap=cmap, point_size=point_size)
    plotter.add_scalar_bar()
    if graph is not None:
        src_g, dst_g, _ = find(graph)
        lines = np.zeros((2*len(src_g), 3))
        for n, (s, d) in enumerate(zip(src_g, dst_g)):
            lines[2 * n, :] = xyz[s]
            lines[2 * n + 1, :] = xyz[d]

        # lines = pv.line_segments_from_points(lines)
        # # Create mapper and add lines
        # mapper = vtk.vtkDataSetMapper()
        # mapper.SetInputData(lines)
        plotter.add_lines(lines, width=1)

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

    return plotter

#
# def mymessage(picked_cells):
#     print(picked_cells.point_arrays['vtkOriginalPointIds'])
#     # print(picked_cells, "Hello World")