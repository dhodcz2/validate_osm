import abc
import collections
import datetime
import functools
from typing import Iterable, Iterator

import numpy as np
import pandas as pd

from validateosm.source import (
    Source,
    SourceOSM,
    data,
    group,
    aggregate,
    enumerative
)
from validate_building_height.building import (
    SourceBuilding, SourceOSMBuilding
)


class Height(SourceBuilding, abc.ABC):
    def __init__(self):
        super(Height, self).__init__()
        self.index_max_height: dict[int, int] = {}
        self.index_max_floors: dict[int, int] = {}

    @data.validate('Float64')
    @abc.abstractmethod
    def floors(self):
        """The number of floors in the building"""

    @data.validate('Float64')
    @abc.abstractmethod
    def height_m(self):
        """The physical height of the building in meters"""

    @aggregate(name='height_m', dtype='Float64')
    def _(self) -> Iterable[float]:
        HEIGHT_TO_FLOOR_RATIO = 3.6
        data = self.groups._data
        data['height_m'].update(
            data.loc[data['height_m'].isna() & data['floors'].notna(), 'floors'] * HEIGHT_TO_FLOOR_RATIO
        )

        yield from self.groups.ungrouped['height_m']
        notna = data[data['height_m'].notna()].index
        for i, gdf in enumerate(self.groups.grouped):
            intersection = gdf.index.intersection(notna)
            if not len(intersection):
                yield np.nan
                continue
            gdf = gdf.loc[intersection]
            valid = zip(gdf.index, gdf['height_m'])
            loc_max, max = next(valid)
            for loc, height in valid:
                if height > max:
                    loc_max = loc
                    max = height
            self.index_max_height[i] = loc_max
            yield max

    @aggregate(name='floors', dtype='Float64')
    def _(self) -> Iterable[float]:
        yield from self.groups.ungrouped['floors']

        data = self.groups._data
        notna = data[data['floors'].notna()].index
        for i, gdf in enumerate(self.groups.grouped):
            intersection = gdf.index.intersection(notna)
            if not len(intersection):
                yield np.nan
                continue
            gdf = gdf[intersection]
            valid = zip(gdf.index, gdf['floors'])
            loc_max, max = next(valid)
            for loc, floors in valid:
                if floors > max:
                    loc_max = loc
                    max = floors
            self.index_max_floors[i] = loc_max
            yield max

    @aggregate(name='timestamp', dtype='datetime64[ns]')
    def _(self) -> Iterator[datetime.datetime]:
        yield from self.groups.ungrouped['timestamp']
        index_max = collections.ChainMap(self.index_max_height, self.index_max_floors)
        yield from (
            gdf.loc[index_max[i], 'timestamp']
            if i in index_max
            else pd.NaT
            for i, gdf in enumerate(self.groups.grouped)
        )

    @aggregate(name='start_date', dtype='datetime64[ns]')
    def _(self) -> Iterator[datetime.datetime]:
        yield from self.groups.ungrouped['start_date']
        index_max = collections.ChainMap(self.index_max_height, self.index_max_floors)
        yield from (
            gdf.loc[index_max[i], 'start_date']
            if i in index_max
            else pd.NaT
            for i, gdf in enumerate(self.groups.grouped)
        )


class NeedlesHeight(SourceOSMBuilding, Height, abc.ABC):
    @enumerative(float)
    def floors(self):
        yield from (
            element.tag('building:levels')
            for element in self.resource
        )

    @enumerative(float)
    def height_m(self):
        yield from (
            element.tag('height')
            for element in self.resource
        )
