from typing import Iterator

from geopandas import GeoDataFrame, GeoSeries
from pandas import Series, DataFrame

from functools import lru_cache

import numpy as np
import pandas as pd

if False:
    from .source import Source


class DescriptorGroups:

    # TODO: Work out a way to exclude the 1-len groups from the aggregation but include them in the aggregate

    def __get__(self, instance: 'Source', owner):
        self._source = instance
        if instance is None:
            return self
        if not hasattr(instance, '_groups'):
            instance._groups = instance.data.groupby(level='group', dropna=True).indices

    @lru_cache(1)
    @property
    def index(self) -> pd.Index:
        return pd.Index(self._source.groups.keys(), name='group')

    @lru_cache(1)
    @property
    def dataframes(self) -> list[GeoDataFrame]:
        data = self._source.data
        return [
            data.iloc[iloc]
            for iloc in self._source.groups.values()
        ]

    def __iter__(self) -> Iterator[GeoDataFrame]:
        yield from self.dataframes

    def __eq__(self, other):
        return self._source is other._source

    def __hash__(self):
        return id(self._source)
