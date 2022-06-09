from distutils.extension import Extension

from Cython.Build import cythonize
from distutils.core import setup

import os
import pyximport
import warnings
import numpy as np

include_dirs = [
    np.get_include(),
    os.path.dirname(__file__),
]

extension = Extension(
    name='cfuncs',
    sources=['cfuncs.pyx'],
    include_dirs=include_dirs,
    define_macros=[('NPY_NO_DEPRECATED_API', 'NPY_1_7_API_VERSION')],
)

setup(
    ext_modules=cythonize(
        extension,
        compiler_directives={'language_level': "3"},
    ),
    name='cfuncs',
)
