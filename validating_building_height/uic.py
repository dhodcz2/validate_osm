import datetime


from ValidateOSM.source import *
from ValidateOSM.source.static import File
# TODO: For some reason if I do not specifically import from source.static instead of source,
#   isinstance(files, File) in static.py will return False instead of True
from validating_building_height.base import Height

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


class EastUicOSM(EastUIC):
    name = 'osm'
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

if __name__ == '__main__':
    # msbf = EastUicMSBF()
    # data = msbf.data
    # data.head()
    osm = EastUicOSM()
    agg = osm.aggregate
    msbf = EastUicMSBF()
    agg = msbf.aggregate
    print()
