from validate_building_height.source import (
    HeightChicagoBuildingFootprints,
    HeightOSM,
    # SourceMSBuildingFootprints
)
from validate_osm.source import BBox

from validate_osm.compare import Compare
from validate_building_height.compare import (
    uic, rio, chicago, loop
)

__all__ = [
    BBox.__name__,
    HeightChicagoBuildingFootprints.__name__,
    HeightOSM.__name__,
    # SourceMSBuildingFootprints.__name__,
]
