from distutils.core import setup, Extension
from Cython.Build import cythonize
from Cython.Distutils import build_ext
import numpy as np

ext_modules = [
    Extension("smutsia.graph._minimum_spanning_tree",
              ["smutsia/graph/_minimum_spanning_tree.pyx"]),
]

setup(
    name="smutsia",
    cmdclass = {'build_ext': build_ext},
    ext_modules = cythonize(ext_modules),
    include_dirs = [np.get_include()]
)
