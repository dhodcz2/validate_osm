from typing import Type

from validate_building_height.source import (
    SourceMSBuildingFootprints,
    HeightOSM,
    HeightChicagoBuildingFootprints
)
from validate_osm.compare.compare import Compare
from validate_osm.source import BBox
from validate_osm.util.compare_partial import partial

uic = BBox([41.86230680163941, -87.65090039861178, 41.87438700655404, -87.64710239060574])
rio = BBox([-23.033529148211027, -43.36733954814765, -22.768918993658822, -43.144602821095134])
chicago = BBox([41.83099018739837, -87.66603456346172, 41.90990199281114, -87.5919345279835])
loop = BBox([41.875558732154204, -87.63817137119877, 41.888683051775764, -87.60959352357703])


# CompareChicago: Type[Compare] = partial(
#     Compare,
#     chicago,
#     HeightOSM, HeightChicagoBuildingFootprints,
#     verbose=True,
# )
#
# CompareUIC: Type[Compare] = partial(
#     Compare,
#     uic,
#     # HeightOSM, HeightChicagoBuildingFootprints, SourceMSBuildingFootprints,
#     HeightOSM, SourceMSBuildingFootprints,
#     verbose=True,
# )
#
# CompareRio: Type[Compare] = partial(
#     Compare,
#     rio,
#     HeightOSM, SourceMSBuildingFootprints,
#     verbose=True,
# )
#
if __name__ == '__main__':
    from validate_osm.source.source import BBox

    bbox = BBox((41.87562863242608, -87.63515367506594, 41.88690149672215, -87.62048834003896), crs='epsg:4326')
    # loop = compare[bbox]
    # loop.plot.difference_scaled('osm', 'cbf', value=['floors', 'height_m'])
    # loop.matrix.percent_difference('osm', 'cbf', ['height_m', 'floors'])
    # loop.results('osm', 'cbf')
    # loop.floc[26]
    # loop.plot.difference_scaled('osm', 'cbf', 'height_m')

"""
# output results to json
validateosm (1 2 3 4) 'height_m' 'floors' output.csv output.json
# push json to REST api 
validateosm commit ./output.json
validateosm --user dhodcz2 --password password  push 
"""
