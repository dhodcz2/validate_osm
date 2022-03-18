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
    HeightOSM, HeightChicagoBuildingFootprints, SourceMSBuildingFootprints,
    bbox=chicago
)

CompareUIC: Type[Compare] = partial(
    Compare,
    # HeightOSM, HeightChicagoBuildingFootprints, SourceMSBuildingFootprints,
    HeightOSM, SourceMSBuildingFootprints,
    bbox=uic
)

CompareRio: Type[Compare] = partial(

    Compare,
    HeightOSM, SourceMSBuildingFootprints,
    bbox=rio
)

# TODO: In compare.data, add
if __name__ == '__main__':
    # compare = CompareUIC(ignore_file=True)
    # compare.data
    # compare.aggregate
    # ...
    compare = CompareChicago(ignore_file=True)
    compare.data
    compare.aggregate
