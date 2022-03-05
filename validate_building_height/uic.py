import datetime

from validateosm.source import *
from validateosm.source.static import File
# TODO: For some reason if I do not specifically import from source.static instead of source,
#   isinstance(files, File) in static.py will return False instead of True
from validate_building_height.base import Height, HeightOSM

import abc


class EastUIC(Height, abc.ABC):
    bbox = BBox([41.86230680163941, -87.65090039861178, 41.87438700655404, -87.64710239060574])


class EastUicMSBF(EastUIC):
    name = 'msbf'
    link = ''
    resource = StaticNaive(
        files=File(path='/home/arstneio/Downloads/chicago-microsoft.feather'),
        crs=4326,
        columns=['geometry']
    )

    def geometry(self):
        print()
        return self.resource['geometry']

    def timestamp(self):
        print()
        return datetime.datetime(2020, 6, 18)

    def address(self):
        ...

    def height_m(self):
        ...

    def start_date(self):
        ...

    def floors(self):
        ...


class EastUicStaticOSM(EastUIC):
    name = 'osm_static'
    link = ''
    resource = StaticNaive(
        files=File(path='/home/arstneio/Downloads/chicago-osm.feather'),
        crs=4326,
        columns=['geometry']
    )

    def geometry(self):
        return self.resource['geometry']

    def timestamp(self):
        ...

    def address(self):
        ...

    def height_m(self):
        ...

    def start_date(self):
        ...

    def floors(self):
        ...


from validateosm.source.source_osm import SourceOSM


class EastUicDynamicOSM(EastUIC, HeightOSM):
    name = 'osm'
    link = ''


if __name__ == '__main__':
    from validateosm.compare.compare import Compare

    # compare = Compare(EastUicStaticOSM, EastUicMSBF, ignore_file=True)
    compare = Compare(EastUicDynamicOSM, EastUicStaticOSM)
    footprints = compare.footprint.footprints
    data = compare.data
    aggregate = compare.aggregate
    print()
