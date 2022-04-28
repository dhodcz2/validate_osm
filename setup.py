from setuptools import find_packages
from setuptools import setup

with open('requirements.txt') as f:
    install_requires = f.read()
setup(
    name='validate_osm',
    description='extracting, comparing, and validating information from OSM',
    author='Daniel Hodczak',
    author_email='dhodcz2@uic.edu',
    packages=find_packages(),
    python_requires='>=3.10',
    install_requires=install_requires,
)
