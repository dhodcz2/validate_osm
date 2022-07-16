# import versioneer

# Skip Cython build if not available
try:
    from Cython.Build import cythonize
except ImportError:
    cythonize = None

import os
from distutils.core import setup
from distutils.extension import Extension

import numpy as np
from Cython.Build import cythonize

# Extension(
#     'util.pfuncs',
#     sources=['util/pfuncs.pyx', 'util/globals.c'],
#     include_dirs=[
#         np.get_include(),
#         os.path.dirname(__file__),
#         os.path.join(os.path.dirname(__file__), 'util'),
#     ],
#     define_macros=[('NPY_NO_DEPRECATED_API', 'NPY_1_7_API_VERSION')],
#     # libraries=['geos_c'],
#     # library_dirs=[os.path.join('/', 'usr', 'lib', find_library('geos_c'))],
# ),
#
# Extension(
#     'util.cfuncs',
#     sources=['util/cfuncs.pyx', 'util/globals.c'],
#     include_dirs=[
#         np.get_include(),
#         os.path.dirname(__file__),
#         os.path.join(os.path.dirname(__file__), 'util'),
#     ],
#     define_macros=[('NPY_NO_DEPRECATED_API', 'NPY_1_7_API_VERSION')],
#     # libraries=['geos_c'],
#     # library_dirs=[os.path.join('/', 'usr', 'lib', find_library('geos_c'))],
# ),
#

ext_modules = [

    Extension(
        'util.ops',
        sources=['util/ops.pyx', 'util/globals.c'],
        include_dirs=[
            np.get_include(),
            os.path.dirname(__file__),
            os.path.join(os.path.dirname(__file__), 'util'),
        ],
        define_macros=[('NPY_NO_DEPRECATED_API', 'NPY_1_7_API_VERSION')],
    ),
    Extension(
        'util.decomposition',
        sources=['util/decomposition.pyx', 'util/globals.c'],
        define_macros=[('NPY_NO_DEPRECATED_API', 'NPY_1_7_API_VERSION')],
        include_dirs=[
            np.get_include(),
            os.path.dirname(__file__),
            os.path.join(os.path.dirname(__file__), 'util'),
        ],
        language='c++',
    ),
]


setup(
    ext_modules=cythonize(
        ext_modules,
        compiler_directives={'language_level': "3"},
    ),
    include_package_data=True,
    name='util',
)
