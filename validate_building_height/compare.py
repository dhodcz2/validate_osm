from functools import partial
from typing import Type

from validate_building_height.source import (
    SourceMSBuildingFootprints,
    HeightOSM,
    HeightChicagoBuildingFootprints
)
from validate_osm.compare.compare import Compare
from validate_osm.source.source import BBox

uic = BBox([41.86230680163941, -87.65090039861178, 41.87438700655404, -87.64710239060574])
rio = BBox([-23.033529148211027, -43.36733954814765, -22.768918993658822, -43.144602821095134])
chicago = BBox([41.83099018739837, -87.66603456346172, 41.90990199281114, -87.5919345279835])

CompareChicago: Type[Compare] = partial(
    Compare,
    chicago,
    HeightOSM, HeightChicagoBuildingFootprints, SourceMSBuildingFootprints,
    verbose=True,
    # debug=True works and contains more detailed information
)

CompareUIC: Type[Compare] = partial(
    Compare,
    uic,
    # HeightOSM, HeightChicagoBuildingFootprints, SourceMSBuildingFootprints,
    HeightOSM, SourceMSBuildingFootprints,
    verbose=True,
)

CompareRio: Type[Compare] = partial(
    Compare,
    rio,
    HeightOSM, SourceMSBuildingFootprints,
    verbose=True
)

# TODO: logger.debug for processes; logger.info for things that may take time.
# TODO: Check that we are loading .feather and not .geojson in this output. Perhaps add an attribute to dataclass:
"""
2022-03-18 03:09:09,059 - INFO - Building Footprints (current).geojson took 5.217855310440063 minutes to load.
2022-03-18 03:09:09,059 - INFO - serializing /home/arstneio/PycharmProjects/ValidateOSM/validate_osm/source/static/StaticNaive/Building Footprints (current).geojson
2022-03-18 03:09:17,296 - INFO - Building Footprints (current).feather to 0.1372891624768575 minutes to serialize.
2022-03-18 03:09:17,297 - INFO - reading Building Footprints (current).feather
2022-03-18 03:09:24,854 - INFO - Building Footprints (current).geojson took 0.1259451150894165 minutes to load."""

if __name__ == '__main__':
    compare = CompareChicago(ignore_file=True)
    compare.data
    compare.aggregate
    print()
