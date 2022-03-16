import abc
import collections

import numpy as np
import pandas as pd

from validate_building_height.building import (
    BuildingSource, BuildingSourceOSM
)
from validateosm.source import (
    data,
    enumerative
)
from validateosm.source.aggregate import AggregateFactory


class HeightAggregateFactory(AggregateFactory):
    def __init__(self, *args, **kwargs):
        super(HeightAggregateFactory, self).__init__(*args, **kwargs)
        self.index_max_height: dict[int, int] = {}
        self.index_max_floors: dict[int, int] = {}

    def __call__(self, *args, **kwargs):
        obj = super(HeightAggregateFactory, self).__call__(*args, **kwargs)
        self.index_max_height.clear()
        self.index_max_floors.clear()
        return obj

    def height_m(self):
        HEIGHT_TO_FLOOR_RATIO = 3.6
        data = self.groups.data
        data['height_m'].update(
            data.loc[data['height_m'].isna() & data['floors'].notna(), 'floors'] * HEIGHT_TO_FLOOR_RATIO
        )

        def gen():
            yield from self.groups.ungrouped['height_m']
            notna = data[data['height_m'].notna()].index
            for i, gdf in enumerate(self.groups.grouped):
                intersection = gdf.index.intersection(notna)
                if not len(intersection):
                    yield np.nan
                else:
                    gdf = gdf.loc[intersection]
                    valid = zip(gdf.index, gdf['height_m'])
                    loc_max, max = next(valid)
                    for loc, height in valid:
                        if height > max:
                            loc_max = loc
                            max = height
                    self.index_max_height[i] = loc_max
                    yield max

        return pd.Series(data=gen(), index=self.groups.index, dtype='Float64')

    def floors(self):

        def gen():
            yield from self.groups.ungrouped['floors']
            data = self.groups.data
            notna = data[data['floors'].notna()].index
            for i, gdf in enumerate(self.groups.grouped):
                intersection = gdf.index.intersection(notna)
                if not len(intersection):
                    yield np.nan
                else:
                    gdf = gdf.loc[intersection]
                    valid = zip(gdf.index, gdf['floors'])
                    loc_max, max = next(valid)
                    for loc, floors in valid:
                        if floors > max:
                            loc_max = loc
                            max = floors
                    self.index_max_floors[i] = loc_max
                    yield max

        return pd.Series(data=gen(), index=self.groups.index, dtype='Float64')

    def timestamp(self):
        def gen():
            yield from self.groups.ungrouped['timestamp']
            index_max = collections.ChainMap(self.index_max_height, self.index_max_floors)
            yield from (
                gdf.loc[index_max[i], 'timestamp',]
                if i in index_max
                else pd.NaT
                for i, gdf in enumerate(self.groups.grouped)
            )

        return pd.Series(data=gen(), index=self.groups.index, dtype='datetime64[ns]')

    def start_date(self):
        def gen():
            yield from self.groups.ungrouped['start_date']
            index_max = collections.ChainMap(self.index_max_height, self.index_max_floors)
            yield from (
                gdf.loc[index_max[i], 'start_date']
                if i in index_max
                else pd.NaT
                for i, gdf in enumerate(self.groups.grouped)
            )

        return pd.Series(data=gen(), index=self.groups.index, dtype='datetime64[ns]')


class Height(BuildingSource, abc.ABC):
    aggregate_factory = HeightAggregateFactory()

    @data.validate('Float64')
    @abc.abstractmethod
    def floors(self):
        """The number of floors in the building"""

    @data.validate('Float64')
    @abc.abstractmethod
    def height_m(self):
        """The physical height of the building in meters"""


# class HeightOSM(BuildingSourceOSM, Height, abc.ABC):
class HeightOSM(BuildingSourceOSM, Height, abc.ABC):
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
