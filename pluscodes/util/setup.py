import numpy as np
import os
from Cython.Build import cythonize
from distutils.core import setup
from distutils.extension import Extension

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
# TODO: How to handle relative importi

setup(
    ext_modules=cythonize(
        extension,
        compiler_directives={'language_level': "3"},
    ),
    name='cfuncs',
)
