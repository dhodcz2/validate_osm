import itertools
import logging
import os
from typing import Generator, Collection, Union, Iterable
from typing import Hashable

import geopandas as gpd
import networkx
import numpy as np
import pandas as pd
import shapely
from annoy import AnnoyIndex
from geopandas import GeoDataFrame
from networkx import connected_components
from shapely.geometry import box
from shapely.geometry.base import BaseGeometry

from validate_osm.logger import logger, logged_subprocess
from validate_osm.source.bbox import BBox

if False:
    from validate_osm.compare.compare import Compare


# TODO: Perhaps the footprints can be accelerated with multiprocessing over the Summer
#   where annoy.get is called for every entry and a a N x 30 matrix is created;
#   because the geometry cannot be multiprocessed, the geometric testing is done in serial

class CallableFootprint:
    def __init__(self, compare: 'Compare'):
        self.compare = compare
        self.path = compare.directory / 'footprint.feather'
        self._gdf = None

    # def __getitem__(self, item: shapely.geometry.Polygon) -> 'CallableFootprint':
    #     footprints = CallableFootprint(self.compare)
    #     gdf = self.gdf
    #     footprints.gdf = gdf[gdf.geometry.intersects(item)]
    #     return footprints

    def __getitem__(
            self,
            item: Union[shapely.geometry.Polygon, int, str, Iterable[int], Iterable[str], gpd.GeoDataFrame]
    ) -> 'CallableFootprint':
        footprints = CallableFootprint(self.compare)
        gdf = self.gdf
        if isinstance(item, BBox):
            projected = item.to_crs(self.gdf.crs)
            footprints.gdf = gdf[gdf.geometry.intersects(projected.ellipsoidal)]
        elif isinstance(item, shapely.geometry.Polygon):
            footprints.gdf = gdf[gdf.geometry.intersects(item)]
        elif isinstance(item, str):
            footprints.gdf = gdf.loc[[item]]
        elif isinstance(item, int):
            footprints.gdf = gdf.iloc[[item]]
        elif isinstance(item, GeoDataFrame):
            polygon = box(*item.to_crs(self.gdf.crs).total_bounds)
            footprints.gdf = gdf[gdf.geometry.intersects(polygon)]
        elif isinstance(item, Iterable):
            it = iter(item)
            first = next(it)
            if isinstance(item, int):
                footprints.gdf = gdf.iloc[[first, *it]]
            elif isinstance(item, str):
                footprints.gdf = gdf.loc[[first, *it]]
            else:
                raise TypeError(first)
        else:
            raise TypeError(item)
        return footprints

    @property
    def gdf(self) -> GeoDataFrame:
        if self._gdf is not None:
            return self._gdf
        if 'footprint' not in self.compare.redo and 'footprints' not in self.compare.redo and self.path.exists():
            with logged_subprocess(f'reading footprints from {self.path}', timed=False):
                footprints = self.gdf = gpd.read_feather(self.path)
                return footprints
        else:
            with logged_subprocess('creating footprints'):
                try:
                    with self as footprints:
                        return footprints
                except RecursionError:
                    _ = self.compare.data
                    return self.gdf  # Try again, now that data has been created

    @gdf.setter
    def gdf(self, gdf: GeoDataFrame):
        gdf['iloc'] = pd.Series(range(len(gdf)), dtype='int32', index=gdf.index)
        self._gdf = gdf

    @gdf.deleter
    def gdf(self):
        del self._gdf

    def __enter__(self) -> GeoDataFrame:
        # If footprints is called before data, there is an infinite recursion
        # TODO: We should accelerate this somehow
        if self.compare not in self.compare.__class__.data.cache:
            raise RecursionError
        data = self.compare.data
        # data = data[['geometry', 'centroid']].copy()
        # data['geometry'] = data['geometry'].to_crs(3857)
        # data['centroid'] = data['centroid'].to_crs(3857)

        # FIrst, aggregate the geometries where relation or way are identical.
        G = networkx.Graph()
        # TODO: Check that this is functional
        for iloc in data.groupby(['name', 'group']).indices.values():
            G.add_nodes_from(iloc)
            G.add_edges_from(zip(iloc[:-1], iloc[1:]))
        groups = connected_components(G)
        grouped: list[Collection[int]] = [
            group for group in groups
            if len(group) > 1
        ]
        ungrouped: set[int] = set(range(len(data))).difference(itertools.chain.from_iterable(grouped))
        ungrouped: list[int] = list(ungrouped)

        def geometry() -> Generator[shapely.geometry.Polygon, None, None]:
            geom = data['geometry']
            yield from geom.iloc[ungrouped]
            for iloc in grouped:
                yield geom.iloc[iloc].unary_union

        def centroid() -> Generator[shapely.geometry.Point, None, None]:
            cent = data['centroid']
            yield from cent.iloc[ungrouped]
            for iloc in grouped:
                yield cent.iloc[iloc].unary_union.centroid

        data = GeoDataFrame({
            'geometry': gpd.GeoSeries(geometry(), crs=data['geometry'].crs),
            'centroid': gpd.GeoSeries(centroid(), crs=data['centroid'].crs)
        })

        # Second, aggregate the geometries where there is an overlap.
        data['area'] = data.area
        data = data.sort_values(by='area', ascending=False)
        footprints = pd.Series(range(len(data)), index=data.index)
        geometries = data['geometry']

        annoy = AnnoyIndex(2, 'euclidean')
        for i, centroid in enumerate(data['centroid']):
            annoy.add_item(i, (centroid.x, centroid.y))  # lower i, higher area
        annoy.build(10)



        for i, (g, a) in enumerate(data[['geometry', 'area']].values):
            for n in annoy.get_nns_by_item(i, 30):
                if i <= n:
                    continue
                geometry: BaseGeometry = geometries.iat[n]
                if not geometry.intersects(g):
                    continue
                if geometry.intersection(g).area / a < .1:
                    continue
                footprints.iat[i] = footprints.iat[n]
                break

        groupby = footprints.groupby(footprints)

        def geometry() -> Generator[shapely.geometry.Polygon, None, None]:
            geom = data['geometry']
            for iloc in groupby.indices.values():
                if len(iloc) == 1:
                    yield geom.iat[iloc[0]]
                else:
                    yield geom.iloc[iloc].unary_union

        def centroid() -> Generator[shapely.geometry.Point, None, None]:
            c = data['centroid']
            for iloc in groupby.indices.values():
                if len(iloc) == 1:
                    yield c.iat[iloc[0]]
                else:
                    yield c.iloc[iloc].unary_union.centroid

        footprints = GeoDataFrame({
            'geometry': gpd.GeoSeries(geometry(), crs=data['geometry'].crs),
            'centroid': gpd.GeoSeries(centroid(), crs=data['centroid'].crs)
        })

        identity = self.identify(footprints)
        footprints = footprints.set_index(identity)
        logger.debug(f'identified footprints with {identity.name=}')

        self.gdf = footprints
        return footprints

    def identify(self, gdf: GeoDataFrame) -> pd.Series:
        """Inherit this class and overwrite to determine how entries will be identified across datasets, eg. UBID"""
        return pd.Series(range(len(gdf)), index=gdf.index, name='i', dtype='Int64')

    def __exit__(self, exc_type, exc_val, exc_tb):
        if exc_type is not None and self.compare.serialize:
            path = self.path
            if not path.parent.exists():
                os.makedirs(path.parent)
            with logged_subprocess(f'serializing footprints to {path}', timed=False):
                self._gdf.to_feather(path)

    @property
    def _annoy(self) -> AnnoyIndex:
        annoy = AnnoyIndex(2, 'euclidean')
        for i, centroid in enumerate(self.gdf['centroid']):
            annoy.add_item(i, (centroid.x, centroid.y))
        annoy.build(50)
        return annoy

    def __call__(self, data: GeoDataFrame) -> GeoDataFrame:
        data['geometry'] = data['geometry'].to_crs(3857)
        data['centroid'] = data['centroid'].to_crs(3857)
        data['area'] = data['geometry'].area
        with logged_subprocess('building annoy', level=logging.DEBUG):
            annoy = self._annoy
        footprints = self.gdf

        def footprint() -> Generator[Hashable, None, None]:
            # Even with n=30, there are 6 unmatched points out of a list of 75 thousand.
            #   Increasing n does not seem to be the best solution. How can we intelligently ensure a match?
            # TODO: We need to configure the searching algorithm so that a match is guaranteed
            for i, (c, a, g) in enumerate(data[['centroid', 'area', 'geometry']].values):
                for n in annoy.get_nns_by_vector((c.x, c.y), 50):
                    match = footprints.iloc[n]
                    geometry: BaseGeometry = match['geometry']
                    if not geometry.intersects(g):
                        continue
                    if geometry.intersection(g).area / a < .5:
                        continue
                    yield match.name
                    break
                else:
                    # TODO: Should we raise an exception if something hasn't been footprinted?
                    yield np.nan

        with logged_subprocess('applying footprint'):
            index = pd.Series(footprint(), index=data.index, name=footprints.index.name)
        names = [footprints.index.name, *data.index.names]
        data = data.set_index(index, drop=False, append=True)
        data = data.reorder_levels(names)
        data = data.sort_index(axis=0)
        return data
