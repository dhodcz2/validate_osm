from typing import Union

import pandas as pd
from geopandas import GeoDataFrame, GeoSeries
from pandas import IndexSlice as idx


class FootprintIlocIndexer:
    def __init__(self):
        ...

    def __getitem__(self, item) -> Union[GeoDataFrame, GeoSeries]:
        if isinstance(item, tuple):
            item, df = item
        else:
            df = 'aggregate'
        from validate_osm.compare.compare import Compare
        compare: Compare = self._instance
        footprints: Union[GeoDataFrame, GeoSeries] = compare.footprints.iloc[item]

        if isinstance(footprints, (GeoDataFrame, pd.DataFrame)):
            index = footprints.index
        elif isinstance(footprints, (GeoSeries, pd.Series)):
            index = footprints.name
        else:
            raise TypeError(footprints)
        if df == 'aggregate':
            return compare.aggregate.loc[idx[index, :], :]
        elif df == 'data':
            return compare.data.loc[idx[index, :, :, :], :]
        else:
            raise ValueError(df)

    def __get__(self, instance, owner):
        self._instance = instance
        self._owner = owner
        return self
