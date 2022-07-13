from ctypes.util import find_library
import builtins
import numpy as np

import logging
import os
import subprocess
import sys
from pathlib import Path

from pkg_resources import parse_version
from setuptools import Extension, find_packages, setup
from setuptools.command.build_ext import build_ext as _build_ext

# import versioneer

# Skip Cython build if not available
try:
    from Cython.Build import cythonize
except ImportError:
    cythonize = None


log = logging.getLogger(__name__)
ch = logging.StreamHandler()
log.addHandler(ch)

MIN_GEOS_VERSION = "3.5"

import numpy as np
import os
from Cython.Build import cythonize
from distutils.core import setup
from distutils.extension import Extension

# def get_geos_config(option):
#     """Get configuration option from the `geos-config` development utility
#
#     The PATH environment variable should include the path where geos-config is
#     located, or the GEOS_CONFIG environment variable should point to the
#     executable.
#     """
#     cmd = os.environ.get("GEOS_CONFIG", "geos-config")
#     try:
#         stdout, stderr = subprocess.Popen(
#             [cmd, option], stdout=subprocess.PIPE, stderr=subprocess.PIPE
#         ).communicate()
#     except OSError:
#         return
#     if stderr and not stdout:
#         log.warning("geos-config %s returned '%s'", option, stderr.decode().strip())
#         return
#     result = stdout.decode().strip()
#     log.debug("geos-config %s returned '%s'", option, result)
#     return result
#
# def get_geos_paths():
#     """Obtain the paths for compiling and linking with the GEOS C-API
#
#     First the presence of the GEOS_INCLUDE_PATH and GEOS_INCLUDE_PATH environment
#     variables is checked. If they are both present, these are taken.
#
#     If one of the two paths was not present, geos-config is called (it should be on the
#     PATH variable). geos-config provides all the paths.
#
#     If geos-config was not found, no additional paths are provided to the extension. It is
#     still possible to compile in this case using custom arguments to setup.py.
#     """
#     include_dir = os.environ.get("GEOS_INCLUDE_PATH")
#     library_dir = os.environ.get("GEOS_LIBRARY_PATH")
#     if include_dir and library_dir:
#         return {
#             "include_dirs": ["util/pygeos/src", include_dir, np.get_include()],
#             "library_dirs": [library_dir],
#             "libraries": ["geos_c"],
#         }
#
#     geos_version = get_geos_config("--version")
#     if not geos_version:
#         log.warning(
#             "Could not find geos-config executable. Either append the path to geos-config"
#             " to PATH or manually provide the include_dirs, library_dirs, libraries and "
#             "other link args for compiling against a GEOS version >=%s.",
#             MIN_GEOS_VERSION,
#         )
#         return {}
#
#     if parse_version(geos_version) < parse_version(MIN_GEOS_VERSION):
#         raise ImportError(
#             "GEOS version should be >={}, found {}".format(
#                 MIN_GEOS_VERSION, geos_version
#             )
#         )
#
#     libraries = []
#     library_dirs = []
#     include_dirs = ["./util/pygeos/src"]
#     extra_link_args = []
#     for item in get_geos_config("--cflags").split():
#         if item.startswith("-I"):
#             include_dirs.extend(item[2:].split(":"))
#
#     for item in get_geos_config("--clibs").split():
#         if item.startswith("-L"):
#             library_dirs.extend(item[2:].split(":"))
#         elif item.startswith("-l"):
#             libraries.append(item[2:])
#         else:
#             extra_link_args.append(item)
#
#     include_dirs.append(np.get_include())
#     include_dirs.append(os.path.join(os.path.dirname(__file__), "util"))
#     include_dirs.append(os.path.dirname(__file__))
#
#     return {
#         "include_dirs": include_dirs,
#         "library_dirs": library_dirs,
#         "libraries": libraries,
#         "extra_link_args": extra_link_args,
#     }
#
# ext_options = get_geos_paths()

# include_dirs = [
#     np.get_include(),
#     os.path.join(os.path.dirname(__file__), "util"),
#     os.path.dirname(__file__),
# ]
ext_modules = [

    Extension(
        'util.pfuncs',
        sources=['util/pfuncs.pyx', 'util/globals.c'],
        include_dirs=[
            np.get_include(),
            os.path.dirname(__file__),
            os.path.join(os.path.dirname(__file__), 'util'),
        ],
        define_macros=[('NPY_NO_DEPRECATED_API', 'NPY_1_7_API_VERSION')],
        # libraries=['geos_c'],
        # library_dirs=[os.path.join('/', 'usr', 'lib', find_library('geos_c'))],
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
        # libraries=['geos_c'],
        # library_dirs=[os.path.join('/', 'usr', 'lib', find_library('geos_c'))],
    ),

    Extension(
        'util.tessellation',
        sources=['util/tessellation.pyx', 'util/globals.c'],
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
