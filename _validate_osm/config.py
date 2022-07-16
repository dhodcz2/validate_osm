from collections import namedtuple
import os
import numbers
from dataclasses import dataclass
from pathlib import Path

# ROOT = Path(__file__).parent
# ABSTRACTION_CACHE = ROOT / 'abstracter' / 'cache'
ROOT = Path(__file__).parent.parent

# TODO: Clean this up

@dataclass
class Config:
    root: str
    cache_file: namedtuple
    join_identifiers: bool
    join_dist_elem: bool
    join_dist: bool
    drop_height_na: bool
    validation_result_cols: list[str]
    haystack_drop_cols: list[str]
    distance_limit_meters: numbers.Real
    join_epsg: str
    debug: bool
    x_size: int
    y_size: int


ValidateOSMConfig = Config(
    # root=os.path.dirname(os.path.abspath(__file__)),
    root=Path(__file__).parent,
    cache_file=namedtuple('CacheFileType', ('extension', 'driver'))('feather', 'feather'),
    join_identifiers=False,
    join_dist=False,
    join_dist_elem=True,
    drop_height_na=True,
    haystack_drop_cols=['osm_id'],
    validation_result_cols='height_hay geometry_hay address_hay validation_by building_id_hay'.split(),
    distance_limit_meters=50,
    join_epsg='epsg:3857',
    debug=False,
    x_size=500,
    y_size=500

)
