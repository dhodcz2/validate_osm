from ctypes.util import find_library
import numpy as np
import os
from Cython.Build import cythonize
from distutils.core import setup
from distutils.extension import Extension

ext_modules = [

    Extension(
        'util.pfuncs',
        sources=['util/pfuncs.pyx', 'util/globals.c'],
        include_dirs=[
            np.get_include(),
            os.path.dirname(__file__),
            # os.path.join(os.path.dirname(__file__), 'util'),
        ],
        define_macros=[('NPY_NO_DEPRECATED_API', 'NPY_1_7_API_VERSION')],
        libraries=['geos_c'],
        library_dirs=[os.path.join('/', 'usr', 'lib', find_library('geos_c'))],
    ),

    Extension(
        'util.cfuncs',
        sources=['util/cfuncs.pyx', 'util/globals.c'],
        include_dirs=[
            np.get_include(),
            os.path.dirname(__file__),
            os.path.join(os.path.dirname(__file__), 'util'),
        ],
        define_macros=[('NPY_NO_DEPRECATED_API', 'NPY_1_7_API_VERSION')],
        libraries=['geos_c'],
        library_dirs=[os.path.join('/', 'usr', 'lib', find_library('geos_c'))],
    ),

    Extension(
        'util.claim',
        sources=['util/cclaim.pyx', 'util/globals.c'],
        include_dirs=[
            np.get_include(),
            os.path.dirname(__file__),
            os.path.join(os.path.dirname(__file__), 'util'),
        ],
        define_macros=[('NPY_NO_DEPRECATED_API', 'NPY_1_7_API_VERSION')],
        language='c++',
        libraries=['geos_c'],
        library_dirs=[os.path.join('/', 'usr', 'lib', find_library('geos_c'))],
    ),

    Extension(
        'util.length',
        sources=['util/cylength.pyx', 'util/globals.c'],
        include_dirs=[
            np.get_include(),
            os.path.dirname(__file__),
            os.path.join(os.path.dirname(__file__), 'util'),
            os.path.join(os.path.dirname(__file__), 'util/pygeos/src'),
        ],
        define_macros=[('NPY_NO_DEPRECATED_API', 'NPY_1_7_API_VERSION')],
        libraries=['geos_c'],
        library_dirs=[os.path.join('/', 'usr', 'lib', find_library('geos_c'))],
    ),

    Extension(
        'util._geos',
        sources=[
            'util/_geos.pyx',
            'util/globals.c',
        ],
        include_dirs=[
            np.get_include(),
            os.path.dirname(__file__),
            os.path.join(os.path.dirname(__file__), 'util'),
            os.path.join(os.path.dirname(__file__), 'util/pygeos/src'),
        ],
        define_macros=[('NPY_NO_DEPRECATED_API', 'NPY_1_7_API_VERSION')],
        libraries=['geos_c'],
        library_dirs=[os.path.join('/', 'usr', 'lib', find_library('geos_c'))],
    ),

]


setup(
    ext_modules=cythonize(
        ext_modules,
        compiler_directives={'language_level': "3"},
    ),
    name='util',
)