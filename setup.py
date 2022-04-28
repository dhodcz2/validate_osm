from distutils.core import setup

setup(
    name='validate_osm',
    description='extracting, comparing, and validating information from OSM',
    author='Daniel Hodczak',
    author_email='dhodcz2@uic.edu',
    packages=['validate_osm', 'streetview_height', 'validate_building_height']
)
