#include <vector>
#include <map>
#include <complex>
#include "points.h"
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>

namespace py = pybind11;

void add_edges(std::vector<int> *a, std::vector<int> *b, std::vector<int> *edges){
    for(auto i = a->begin(); i < a->end(); i++){
        for(auto j = b->begin(); j < b->end(); j++){
            edges->push_back(*i);
            edges->push_back(*j);
        }
    }
}

py::array build_spherical_edges(py::array_t<int> lidx_ndarray, py::array_t<double> points_ndarray, int height, int width){
    py::buffer_info lidx_buff = lidx_ndarray.request(), points_buff = points_ndarray.request();

    int *lidx = (int *) lidx_buff.ptr;
    // check input dimensions
    if ( points_ndarray.ndim() != 2 )
        throw std::runtime_error("Input should be 2-D NumPy array");
    if ( points_ndarray.shape()[1] != 3 )
        throw std::runtime_error("Input should have size [N,3]");

    py::print("Point cloud shape: ");
    py::print(points_ndarray.shape()[0], points_ndarray.shape()[1]);

    // allocate std::vector (to pass to the C++ function)
    std::vector<PointXYZ> points(points_ndarray.shape()[0]);

    for (int i = 0; i < points_ndarray.shape()[0]; i++){
        points[i].x = points_ndarray.data()[3*i];
        points[i].y = points_ndarray.data()[3*i + 1];
        points[i].z = points_ndarray.data()[3*i + 2];
    }

    py::print("Point array shape: ", points.size());
    // initialise map pix ->
    std::map<int, std::vector<int>> pxl_map;
    std::vector<int> touched_pxl(height*width, 0);
    py::print(touched_pxl.size(), height*width);
    for (auto idx = 0; idx < lidx_buff.shape[0]; idx++){
        if (pxl_map.find(lidx[idx]) != pxl_map.end()) { // if the key exists we add another element
            pxl_map[lidx[idx]].push_back(idx);
        }
        else{ // otherwise we add a new key to the map
            std::vector<int> first_el;
            first_el.push_back(idx);
            pxl_map.insert(std::pair<int, std::vector<int>>(lidx[idx], first_el));
        }
        touched_pxl[lidx[idx]] = 1;
    }
    std::vector<int> edges;
    std::vector<bool> auto_connected(height*width, false);
    // iterating over points
    for (auto idx = 0; idx < points.size(); idx++){

        int pxl = lidx[idx];
        int i = (int) pxl / width, j = pxl % width;
        int right_neigh, bottom_neigh;
        int skip_connection = 0;
        bool right_connected = false;
        PointXYZ delta;
        bottom_neigh = (i + 1) * width + j;
        // add connections with bottom neighbor
        if (pxl_map.find(bottom_neigh) != pxl_map.end()){
            // bottom neighbor exist
            for( auto k = pxl_map.find(bottom_neigh)->second.begin(); k < pxl_map.find(bottom_neigh)->second.end(); k++){
                delta = points[idx] - points[*k];
                if (std::abs(delta.z) < 0.50 && delta.l2_norm() < 5.0){
                    edges.push_back(idx);
                    edges.push_back(*k);
                }
            }
        }
        // if right pixel is empty we shift by one until we find a non empty pixel
        while( !right_connected && skip_connection < 5){
            right_neigh = (i * width) + ((j + 1) % width);
            if(touched_pxl[right_neigh] > 0){
                for( auto k = pxl_map.find(right_neigh)->second.begin(); k < pxl_map.find(right_neigh)->second.end(); k++) {
                    delta = points[idx] - points[*k];
                    if (delta.l2_norm() < 1.0 && std::abs(delta.z) < 0.50) {
                        right_connected = true;
                        edges.push_back(idx);
                        edges.push_back(*k);
                    }
                }
            }
            j = right_neigh % width;
            skip_connection++;
        }
        // auto connections
        if(! auto_connected[pxl] && pxl_map.find(idx)->second.size() > 1){
            for(auto k = pxl_map.find(idx)->second.begin(); k < pxl_map.find(idx)->second.end(); k++){
                for(auto h = k; h < pxl_map.find(idx)->second.end(); h++){
                    if (h != k){
                        delta = points[*k] - points[*h];
                        if (delta.l2_norm() < 0.5 && std::abs(delta.z) < 0.5) {
                            right_connected = true;
                            edges.push_back(*h);
                            edges.push_back(*k);
                        }
                    }
                }
            }
            auto_connected[pxl] = true;
        }
    }

    ssize_t              ndim    = 2;
    std::vector<ssize_t> shape   = { (int) edges.size() / 2 , 2 };
    std::vector<ssize_t> strides = { sizeof(int)*2 , sizeof(int) };

    // return 2-D NumPy array
    return py::array(py::buffer_info(
            edges.data(),                           /* data as contiguous array  */
            sizeof(int),                            /* size of one scalar        */
            py::format_descriptor<int>::format(),   /* data type                 */
            ndim,                                    /* number of dimensions      */
            shape,                                   /* shape of the matrix       */
            strides                                  /* strides for each axis     */
    ));

}

PYBIND11_MODULE(spherical_edges, m) {
    m.def("build_spherical_edges",
            &build_spherical_edges,
            "Generate edges for the graph using info of projection by layers",
            py::arg("lidx"),
            py::arg("points"),
            py::arg("height")=64,
            py::arg("width")=2048);
}