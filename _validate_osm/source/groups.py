import dataclasses
import functools
import itertools
from typing import Collection

import pandas as pd
from geopandas import GeoDataFrame

if False:
    pass


@dataclasses.dataclass
class Groups:
    data: GeoDataFrame = dataclasses.field(repr=False)
    iloc_grouped: Collection[Collection[int]]
    iloc_ungrouped: Collection[int]

    @functools.cached_property
    def grouped(self) -> list[GeoDataFrame]:
        return [
            self.data.iloc[iter(iloc)]
            for iloc in self.iloc_grouped
        ]

    @functools.cached_property
    def ungrouped(self) -> GeoDataFrame:
        return self.data.iloc[iter(self.iloc_ungrouped)]

    @functools.cached_property
    def index(self) -> pd.MultiIndex:
        identity: str = self.data.index.names[0]
        ungrouped: pd.MultiIndex = self.ungrouped.index
        # index_ungrouped = zip(ungrouped.get_level_values('ubid'), ungrouped.get_level_values('name'))
        index_ungrouped = zip(ungrouped.get_level_values(identity), ungrouped.get_level_values('name'))
        ilocs = [
            next(iter(iloc))
            for iloc in self.iloc_grouped
        ]
        grouped: pd.MultiIndex = self.data.iloc[ilocs].index
        # index_grouped = zip(grouped.get_level_values('ubid'), grouped.get_level_values('name'))
        index_grouped = zip(grouped.get_level_values(identity), grouped.get_level_values('name'))
        return pd.MultiIndex.from_tuples(itertools.chain(index_ungrouped, index_grouped), names=[identity, 'name'])
