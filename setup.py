from distutils.core import  setup

setup(
    name='validateosm',
    version='0.1.0',
    author='Daniel Hodczak',
    author_email='dhodcz2@uic.edu',
    packages=[
        'validateosm', 'validateosm.source', 'validateosm.compare', 'validate_building_height'
    ],
    url='https://github.com/dhodcz2/ValidateOSM',
    description='Global, open validation of databases, particularly that of OpenStreetMaps',
)