import numpy as np
import os
from Cython.Build import cythonize
from distutils.core import setup
from distutils.extension import Extension

ext_modules = [
    Extension(
        'util.pfuncs',
        sources=['util/pfuncs.pyx'],
        include_dirs=[
            np.get_include(),
            os.path.dirname(__file__),
            # os.path.join(os.path.dirname(__file__), 'util'),
        ],
        define_macros=[('NPY_NO_DEPRECATED_API', 'NPY_1_7_API_VERSION')],
    ),
    Extension(
        'util.cfuncs',
        sources=['util/cfuncs.pyx'],
        include_dirs=[
            np.get_include(),
            os.path.dirname(__file__),
            os.path.join(os.path.dirname(__file__), 'util'),
        ],
        define_macros=[('NPY_NO_DEPRECATED_API', 'NPY_1_7_API_VERSION')],
    ),
    Extension(
        'util.claim',
        sources=['util/cclaim.pyx'],
        include_dirs=[
            np.get_include(),
            os.path.dirname(__file__),
            os.path.join(os.path.dirname(__file__), 'util'),
        ],
        define_macros=[('NPY_NO_DEPRECATED_API', 'NPY_1_7_API_VERSION')],
        language='c++',
    ),
    Extension(
        'util.length',
        sources=['util/cylength.pyx'],
        include_dirs=[
            np.get_include(),
            os.path.dirname(__file__),
            os.path.join(os.path.dirname(__file__), 'util'),
            os.path.join(os.path.dirname(__file__), 'util/pygeos/src'),
        ],
        define_macros=[('NPY_NO_DEPRECATED_API', 'NPY_1_7_API_VERSION')],
        # language='c++',
    ),
    Extension(
        'util._geos',
        sources=['util/_geos.pyx'],
        include_dirs=[
            np.get_include(),
            os.path.dirname(__file__),
            os.path.join(os.path.dirname(__file__), 'util'),
        ],
        define_macros=[('NPY_NO_DEPRECATED_API', 'NPY_1_7_API_VERSION')],
        # language='c++',
    )
    # Extension(
    #     'util.test',
    #     sources=['util/test.pyx'],
    #     include_dirs=[
    #         np.get_include(),
    #         os.path.dirname(__file__),
    #         os.path.join(os.path.dirname(__file__), 'util'),
    #         os.path.join(os.path.dirname(__file__), 'util/pygeos/src'),
    #         os.path.join(os.path.dirname(__file__), 'util/pygeos'),
    #     ],
    #     define_macros=[('NPY_NO_DEPRECATED_API', 'NPY_1_7_API_VERSION')],
    # ),
]

setup(
    ext_modules=cythonize(
        ext_modules,
        compiler_directives={'language_level': "3"},
    ),
    name='util',
)
