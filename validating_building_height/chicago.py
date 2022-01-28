import argparse
import datetime
import os.path
from pathlib import Path

import shapely.geometry
from ValidateOSM.config import ROOT
import geopandas as gpd
import abc
from typing import Iterable, Iterator, Union
from pandas import Series, DataFrame
from geopandas import GeoDataFrame, GeoSeries

import numpy as np
import pandas as pd
import shapely
from geopandas import GeoDataFrame
from ValidateOSM.source import (
    data,
    group,
    aggregate,
    Source,
    SourceOSM,
    BBox
)
from ValidateOSM.source import (
    Source, SourceOSM, data, group, aggregate, enumerative, Static
)
from validating_building_height.base import (
    Height, NeedlesHeight,
)


class Chicago(Height, abc.ABC):
    @data.identifier('Int64')
    @abc.abstractmethod
    def bldg_id(self) -> Iterable[int]:
        """The Chicago Building ID"""

    bbox = BBox([41.83099018739837, -87.66603456346172, 41.90990199281114, -87.5919345279835])


class ChicagoBuildingHeightNeedles(NeedlesHeight, Chicago):
    @enumerative(int)
    def bldg_id(self) -> Iterable[int]:
        yield from (
            element.tag('chicago:building_id')
            for element in self.source
        )


class ChicagoBuildingFootprints(Chicago, Height):
    # TODO: Instead of 6 lines of code for classmethod property strings, perhaps just define them at the top
    #   and enforce definition for non-ABCs
    name = 'cbf'
    link = 'https://data.cityofchicago.org/Buildings/Building-Footprints-current-/hz9b-7nh8'
    static = Static(url='https://data.cityofchicago.org/api/geospatial/hz9b-7nh8?method=export&format=GeoJSON')

    @property
    def raw(self) -> GeoDataFrame:
        return self.static

    def geometry(self) -> Iterable[shapely.geometry.base.BaseGeometry]:
        return self.raw.geometry.reset_index(drop=True)

    def address(self) -> Iterable[object]:
        def generator():
            for num, dir, name, type in self.raw.loc[:, ['t_add1', 'pre_dir1', 'st_name1', 'st_type1']]:
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
            for floors in self.raw['stories']
        )

    def bldg_id(self) -> Iterable[int]:
        return self.raw['bldg_id'].reset_index(drop=True)

    def height_m(self) -> Iterable[float]:
        return np.nan

    def timestamp(self) -> Iterable[datetime.datetime]:
        return self.raw['date_bld_2'].reset_index(drop=True)

    def start_date(self) -> Iterable[datetime.datetime]:
        yield from (
            pd.NaT if year == 0
            else datetime.datetime(int(year), 1, 1)
            for year in self.raw['year_built']
        )
