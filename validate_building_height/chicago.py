import abc
import datetime
import functools
from typing import Iterable

import numpy as np
import pandas as pd
import shapely.geometry
from geopandas import GeoDataFrame

from validate_building_height.base import (
    Height,
    HeightOSM
)
from validateosm.source import (
    BBox
)
from validateosm.source import (
    data, enumerative,
)
from validateosm.source.static import (
    StaticNaive, File
)


class Chicago(Height, abc.ABC):
    @data.identifier('Int64')
    @abc.abstractmethod
    def bldg_id(self) -> Iterable[int]:
        """The Chicago Building ID"""

    bbox = BBox([41.83099018739837, -87.66603456346172, 41.90990199281114, -87.5919345279835])


class ChicagoBuildingHeightOSM(HeightOSM, Chicago):
    @enumerative(int)
    def bldg_id(self) -> Iterable[int]:
        yield from (
            element.tag('chicago:building_id')
            for element in self.resource
        )


class ChicagoBuildingFootprints(Chicago, Height):
    # TODO: Instead of 6 lines of code for classmethod property strings, perhaps just define them at the top
    #   and enforce definition for non-ABCs
    name = 'cbf'
    link = 'https://data.cityofchicago.org/Buildings/Building-Footprints-current-/hz9b-7nh8'
    resource = StaticNaive(
        files=File(url='https://data.cityofchicago.org/api/geospatial/hz9b-7nh8?method=export&format=GeoJSON')
    )

    @functools.cached_property
    def resource(self) -> GeoDataFrame:
        return self.resource

    def geometry(self) -> Iterable[shapely.geometry.base.BaseGeometry]:
        return self.resource.geometry.reset_index(drop=True)

    def address(self) -> Iterable[object]:
        def generator():
            for num, dir, name, type in self.resource.loc[:, ['t_add1', 'pre_dir1', 'st_name1', 'st_type1']]:
                if not (name and type):
                    yield None
                    continue
                res: str = num
                for frag in (dir, name, type):
                    if frag is not None:
                        res += f' {frag}'
                yield res

        yield from generator()

    def floors(self) -> Iterable[float]:
        yield from (
            np.nan if floors == 0
            else floors
            for floors in self.resource['stories']
        )

    def bldg_id(self) -> Iterable[int]:
        return self.resource['bldg_id'].reset_index(drop=True)

    def height_m(self) -> Iterable[float]:
        return np.nan

    def timestamp(self) -> Iterable[datetime.datetime]:
        return self.resource['date_bld_2'].reset_index(drop=True)

    def start_date(self) -> Iterable[datetime.datetime]:
        yield from (
            pd.NaT if year == 0
            else datetime.datetime(int(year), 1, 1)
            for year in self.resource['year_built']
        )
