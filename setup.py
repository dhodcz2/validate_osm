
from setuptools import dist
dist.Distribution().fetch_build_eggs(['Cython>=0.15.1', 'numpy>=1.10'])
from Cython.Build import cythonize
import warnings
from pathlib import Path
import numpy
from setuptools import find_packages
from setuptools import setup

path = Path(__file__).parent
with open(path / 'requirements.txt') as f:
    lines = f.readlines()
    # install_requires = f.read()

warnings.warn("install gdal to your system. apparently it rambles about needing gdal-config or something, idk")

# git+https://github.com/pnnl/buildingid-py
# git+https://github.com/pnnl/buildingid-py.git#egg=pnnl-buildingid
install_requires = [
    line.replace('\n', '')
    for line in lines
]

install_requires.append('pnnl-buildingid @ git+https://github.com/pnnl/buildingid-py.git')
setup(
    name='validate_osm',
    description='extracting, comparing, and validating information from OSM',
    author='Daniel Hodczak',
    author_email='dhodcz2@uic.edu',
    packages=find_packages(),
    python_requires='>=3.10',
    install_requires=install_requires,
    # ext_modules=cythonize('streetview_height/cdistance.pyx'),
    include_dirs=[numpy.get_include()],
    ext_modules=cythonize('shadow/cutil/functions.pyx')
)