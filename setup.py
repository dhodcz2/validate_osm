from distutils.core import  setup

setup(
    name='ValidateOSM',
    version='0.1.0',
    author='Daniel Hodczak',
    author_email='dhodcz2@uic.edu',
    packages=['validateosm', 'validateosm.validate_building_height'],
    url='https://github.com/dhodcz2/ValidateOSM',
    description='Global, open validation of databases, particularly that of OpenStreetMaps',
)