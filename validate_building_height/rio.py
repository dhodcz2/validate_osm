import datetime

from validateosm.source import *
from validateosm.source.static import File
from validate_building_height.base import Height, HeightOSM
from validate_building_height.base import (
    Height, HeightOSM
)
import abc


class HeightRioDeJaneiro(Height):
    bbox = BBox([
        -23.033529148211027, -43.36733954814765, -22.768918993658822, -43.144602821095134
    ])


class RioDeJaneiroMSBF(HeightRioDeJaneiro):
    name = 'msbf'
    link = ''

    from validate_building_height.regional import MSBuildingFootprints
    resource = MSBuildingFootprints()

    def geometry(self):
        return self.resource['geometry']

    def timestamp(self):
        return datetime.datetime(2020, 6, 10)

    def address(self):
        ...

    def height_m(self):
        ...

    def start_date(self):
        ...

    def floors(self):
        ...


class RioDeJaneiroOSM(HeightOSM, HeightRioDeJaneiro):
    name = 'osm'
    link = ''


if __name__ == '__main__':
    from validateosm.compare.compare import Compare
    compare = Compare(RioDeJaneiroMSBF, RioDeJaneiroOSM, ignore_file=True)
    data = compare.data
    footprints = compare.footprint.footprints
    aggregate = compare.aggregate
