import warnings
from typing import Type

warnings.filterwarnings('ignore', '.*Shapely GEOS.*', )
from validate_osm import BBox, Compare
from validate_osm.util.compare_partial import partial

uic = BBox([41.86230680163941, -87.65090039861178, 41.87438700655404, -87.64710239060574])
rio = BBox([-23.033529148211027, -43.36733954814765, -22.768918993658822, -43.144602821095134])
chicago = BBox([41.83099018739837, -87.66603456346172, 41.90990199281114, -87.5919345279835])
loop = BBox([41.875558732154204, -87.63817137119877, 41.888683051775764, -87.60959352357703])
champaign = BBox([40.051867168545456, -88.33906821670959, 40.173800444161266, -88.15061302949474])
san_francisco = BBox([37.70396357871276, -122.52904260934639, 37.83856981666945, -122.34570826971019])
manhattan = BBox([40.69859257332866, -74.02316812897904, 40.74358074985354, -73.96698991950707])

from validate_building_height.source import (
    HeightMicrosoftBuildingFootprints as MSBF,
    HeightMicrosoftBuildingFootprints2017 as MSBF2017,
    HeightOpenCityModel as OCM,
    HeightOSM as OSM
)
from validate_building_height.new_york import (
    NewYork3DModel as TDM,
    NewYorkLOD as LOD,
)

# TODO: Perhaps implement scheduling such that OSM downloads are made when there is minimal system load.

CompareChicago: Type[Compare] = partial(Compare, chicago, MSBF, OCM, OSM, verbose=True)
CompareUIC: Type[Compare] = partial(Compare, uic, MSBF, OCM, OSM, verbose=True)
# CompareChampaign: Type[Compare] = partial(Compare, champaign, MSBF, MSBF2017, OCM, OSM, verbose=True)
CompareChampaign: Type[Compare] = partial(Compare, champaign, MSBF, MSBF2017, OCM, OSM, verbose=True)
CompareRio: Type[Compare] = partial(Compare, rio, MSBF, MSBF2017, OSM, verbose=True)
CompareSanFrancisco: Type[Compare] = partial(Compare, san_francisco, MSBF, MSBF2017, OSM, OCM, verbose=True)
CompareManhattan: Type[Compare] = partial(Compare, manhattan, MSBF, TDM, LOD)

# TODO: Memory limitation
if __name__ == '__main__':
    compare: Compare = CompareChampaign(redo=['ocm', 'msbf2017'])
    files = compare.sources['msbf2017'].resource[compare.bbox]
    files
