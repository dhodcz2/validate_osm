from distutils.core import setup

with open('requirements.txt') as f:
    required = f.read().splitlines()

setup(
    name='validate_osm',
    author='Daniel Hodczak',
    author_email='dhodcz2@uic.edu',
    packages=[
        'validate_osm', 'validate_osm.source', 'validate_osm.compare', 'validate_building_height', 'validate_osm.util'
    ],
    url='https://github.com/dhodcz2/ValidateOSM',
    description='Global, open validation of databases, particularly that of OpenStreetMaps',
    install_required=required
)