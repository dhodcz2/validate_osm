import numpy as np
import os
from Cython.Build import cythonize
from distutils.core import setup
from distutils.extension import Extension

ext_modules = [
    Extension(
        'pfuncs',
        sources=['pfuncs.pyx'],
        include_dirs=[np.get_include(), os.path.dirname(__file__)],
        define_macros=[('NPY_NO_DEPRECATED_API', 'NPY_1_7_API_VERSION')],
    ),
    Extension(
        'cfuncs',
        sources=['cfuncs.pyx'],
        include_dirs=[np.get_include(), os.path.dirname(__file__)],
        define_macros=[('NPY_NO_DEPRECATED_API', 'NPY_1_7_API_VERSION')],

    ),
]

setup(
    ext_modules=cythonize(
        ext_modules,
        compiler_directives={'language_level': "3"},
    ),
    name='util',
)
