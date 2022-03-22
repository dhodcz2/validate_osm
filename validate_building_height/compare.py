from typing import Type

from validate_building_height.source import (
    SourceMSBuildingFootprints,
    HeightOSM,
    HeightChicagoBuildingFootprints
)
from validate_osm.compare.compare import Compare
from validate_osm.source.source import BBox
from validate_osm.util.compare_partial import partial


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
    verbose=True,
)

# TODO: Still some situations where multiple Compare.logger handlers
# TODO: Compare[bbox] returns a smaller Compare instance

if __name__ == '__main__':
    # compare = CompareChicago(debug=True, redo=['data', 'footprint', ])
    # compare = CompareChicago(debug=True, redo=['data', 'footprint', ])
    # compare = CompareChicago(debug=True, redo=['data', 'footprint', 'aggregate'])
    # compare.data
    # compare.aggregate
    # print()
    compare = CompareChicago(debug=True, redo=None)
    from validate_osm.source.source import BBox
    smaller = compare[
        BBox([41.874244361608234, -87.63580711113813, 41.88816184961424, -87.61128776307615], crs='epsg:4326')
    ]
    smaller.plot.matches()
    # compare.data
    # compare.footprints
    # compare.plot.matches()
