import functools
import itertools
import logging
from typing import Callable, Collection, Iterator

import networkx
import pandas as pd
import shapely.geometry
from geopandas import GeoDataFrame, GeoSeries
from networkx import connected_components

from validate_osm.source.groups import Groups
from validate_osm.util.scripts import logged_subprocess


class AggregateFactory:

    def __init__(self, compare: object):
        from validate_osm.compare.compare import Compare
        compare: Compare
        self._compare = compare
        self.data = compare.data

    def _decorator(self, func: Callable):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            with logged_subprocess(self._compare.logger, f'{func.__name__}', logging.DEBUG):
                return func(*args, **kwargs)

        return wrapper

    @functools.cached_property
    def _groups(self) -> Groups:
        data = self._compare.data

        def gen():
            for name in data.index.unique('name'):
                df: GeoDataFrame = data.xs(name, level='name')
                yield from (
                    df.loc[loc, 'iloc'].values
                    for loc in df.groupby(level='group', dropna=True).groups.values()
                    if len(loc) > 1
                )
                yield from (
                    df.loc[loc, 'iloc'].values
                    for loc in df.groupby(level='ubid').groups.values()
                    if len(loc) > 1
                )

        G = networkx.Graph()
        for group in gen():
            G.add_nodes_from(group)
            G.add_edges_from(zip(group[:-1], group[1:]))

        iloc_grouped: Collection[Collection[int]] = list(connected_components(G))
        iloc_ungrouped = set(data['iloc']).difference(itertools.chain.from_iterable(iloc_grouped))
        return Groups(data, iloc_grouped, iloc_ungrouped)

    # def __call__(self, groups: Groups):
    #     self.groups = groups
    #     functions = [
    #         (name, func)
    #         for cls in self.__class__.mro()
    #         for name, func in cls.__dict__.items()
    #         if not name.startswith('_')
    #            and isinstance(func, Callable)
    #     ]
    #
    #     result = GeoDataFrame({
    #         name: self._decorator(func)(groups)
    #         for name, func in functions
    #     })
    #     result = result.sort_index(axis=0)
    #     result['iloc'] = range(len(result))
    #     result['geometry'] = result['geometry'].to_crs(3857)
    #     result['centroid'] = result['centroid'].to_crs(3857)
    #     return result

    def __enter__(self) -> GeoDataFrame:
        functions = [
            (name, func)
            for cls in self.__class__.mro()
            for name, func in cls.__dict__.items()
            if not name.startswith('_')
               and isinstance(func, Callable)
        ]
        result = GeoDataFrame({
            name: self._decorator(func)(self)
            for name, func in functions
        })
        result = result.sort_index(axis=0)
        result['iloc'] = range(len(result))
        inval = result[result['geometry'].isna() | result['centroid'].isna()].index
        if len(inval):
            self._compare.logger.warning(f'no geom: {inval}')
            result = result[result['geometry'].notna()]
        return result

    def __exit__(self, exc_type, exc_val, exc_tb):

        ...

    def geometry(self):
        def data():
            yield from self._groups.ungrouped['geometry']
            yield from (
                gdf['geometry'].unary_union
                for gdf in self._groups.grouped
            )

        return GeoSeries(data=data(), index=self._groups.index, crs=3857)

    def centroid(self):
        def data():
            yield from self._groups.ungrouped['centroid']
            yield from (
                gdf['centroid'].unary_union.centroid
                for gdf in self._groups.grouped
            )

        return GeoSeries(data=data(), index=self._groups.index, crs=3857)

    def ref(self):
        single = self._groups.ungrouped['ref']
        # temp = self._groups.data.copy()
        # self._groups.data = temp.assign(centroid=temp['centroid'].to_crs(4326))
        # TODO: This is perhaps suboptimal because calling several to_crs seems inefficient
        centroids: Iterator[shapely.geometry.Point] = (
            gdf['centroid'].to_crs(4326).unary_union.centroid
            for gdf in self._groups.grouped
        )
        multi: Iterator[str] = (
            f'{centroid.y:.4f}\n{centroid.x:.4f}'
            if centroid is not None else ''
            for centroid in centroids
        )
        series = pd.Series(
            itertools.chain(single, multi),
            index=self._groups.index,
            dtype='string'
        )
        # self._groups.data = temp
        return series

