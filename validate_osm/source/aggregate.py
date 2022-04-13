import pandas as pd
import functools
import itertools
import logging
from typing import Callable, Collection, Iterator

import networkx
import shapely.geometry
from geopandas import GeoDataFrame, GeoSeries
from networkx import connected_components
from pandas import Series

from validate_osm.source.groups import Groups
from validate_osm.logger import logger, logged_subprocess

if False:
    from validate_osm.compare.compare import Compare


class FactoryAggregate:
    compare: 'Compare'

    def __init__(self):
        self.functions: list[tuple[str, Callable[[FactoryAggregate], Series]]] = [
            (name, func)
            for cls in self.__class__.mro()
            for name, func in cls.__dict__.items()
            if not name.startswith('_')
               and isinstance(func, Callable)
        ]

    def __hash__(self):
        return hash(self.compare)

    def __eq__(self, other):
        return self.compare == other

    @property
    @functools.lru_cache()
    def _data(self) -> GeoDataFrame:
        return self.compare.data

    def _decorator(self, func: Callable):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            with logged_subprocess(f'{func.__name__}', logging.DEBUG):
                return func(*args, **kwargs)

        return wrapper

    def __iter__(self):
        data = self._data
        for name in data.index.unique('name'):
            gdf: GeoDataFrame = data.xs(name, level='name')
            for loc in gdf.groupby(level='group', dropna=True).groups.values():
                if len(loc) > 1:
                    yield gdf.loc[loc, 'iloc'].values
            for loc in gdf.groupby(level=self.compare.identity).groups.values():
                if len(loc) > 1:
                    yield gdf.loc[loc, 'iloc'].values

    @property
    @functools.lru_cache()
    def _groups(self) -> Groups:
        G = networkx.Graph()
        for group in self:
            G.add_nodes_from(group)
            G.add_edges_from(zip(group[:-1], group[1:]))
        grouped: Collection[Collection[int]] = list(connected_components(G))
        ungrouped = set(self._data['iloc']).difference(itertools.chain.from_iterable(grouped))
        return Groups(self._data, grouped, ungrouped)

    def __call__(self, compare: 'Compare') -> GeoDataFrame:
        self.compare = compare
        # groups = self._groups
        # TODO: Blackpill is that we are calling with the wrong instance
        result = GeoDataFrame({
            name: self._decorator(func)(self)
            for name, func in self.functions
        })
        result = result.sort_index(axis=0)
        result['iloc'] = range(len(result))
        inval = result[
            result['geometry'].isna() | result['centroid'].isna()
            ].index
        if len(inval):
            logger.warning(f'no geom: {inval}')
            result = result[result['geometry'].notna()]
        return result

    def geometry(self):
        data = itertools.chain(
            self._groups.ungrouped['geometry'], (
                gdf['geometry'].unary_union
                for gdf in self._groups.grouped
            )
        )
        return GeoSeries(data, index=self._groups.index, crs=3857)

    def centroid(self):
        data = itertools.chain(
            self._groups.ungrouped['centroid'], (
                gdf['centroid'].unary_union.centroid
                for gdf in self._groups.grouped
            )
        )
        return GeoSeries(data, index=self._groups.index, crs=3857)

    def ref(self):
        single = self._groups.ungrouped['ref']
        centroids: Iterator[shapely.geometry.Point] = (
            gdf['centroid'].to_crs(4326).unary_union.centroid
            for gdf in self._groups.grouped
        )
        multi: Iterator[str] = (
            f'{centroid.y:.4f}, {centroid.x:.4f}'
            if centroid is not None else ''
            for centroid in centroids
        )
        return pd.Series(itertools.chain(single, multi), index=self._groups.index, dtype='string')
