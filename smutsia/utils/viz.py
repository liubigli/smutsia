import numpy as np
import pyvista as pv
from scipy.sparse import find

def plot_cloud(xyz, scalars=None, cmap=None, point_size=1.0, graph=None):

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

    print("TEST FUNC")
    plotter.enable_cell_picking(through=False, callback=mymessage)
    return plotter


def mymessage(picked_cells):
    print(picked_cells.point_arrays['vtkOriginalPointIds'])
    # print(picked_cells, "Hello World")

def analyse_plot(xyz, scalars=None, cmap=None, point_size=1.0, graph=None):
    plotter = plot_cloud(xyz, scalars=scalars, cmap=cmap, point_size=point_size, graph=graph)
    # sel_points = None

    # while True:
    #     if plotter.picked_cells is not None:
    #         pass
