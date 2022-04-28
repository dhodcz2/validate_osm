from setuptools import find_packages
from setuptools import setup

setup(
    name='validate_osm',
    description='extracting, comparing, and validating information from OSM',
    author='Daniel Hodczak',
    author_email='dhodcz2@uic.edu',
    packages=find_packages(),
    python_requires='>=3.10',
    # packages=['validate_osm', 'streetview_height', 'validate_building_height'],
    # py_modules=
)
