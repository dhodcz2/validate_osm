import functools
import itertools
import logging
import os
from typing import Generator, Collection
from typing import Hashable

import geopandas as gpd
import networkx
import numpy as np
import pandas as pd
import shapely
from annoy import AnnoyIndex
from geopandas import GeoDataFrame
from networkx import connected_components
from shapely.geometry.base import BaseGeometry

from validate_osm.util.scripts import logged_subprocess

# TODO: Perhaps the footprints can be accelerated with multiprocessing over the Summer

class CallableFootprint:
    def __init__(self, compare: object):
        from validate_osm.compare.compare import Compare
        compare: Compare
        self.compare = compare
        self.path = compare.directory / 'footprint.feather'
        self._gdf = None

    def __getitem__(self, item: shapely.geometry.Polygon) -> 'CallableFootprint':
        footprints = CallableFootprint(self.compare)
        gdf = self.gdf
        footprints.gdf = gdf[gdf.geometry.intersects(item)]
        return footprints

    @property
    def gdf(self) -> GeoDataFrame:
        if self._gdf is not None:
            return self._gdf
        if 'footprint' not in self.compare.redo and 'footprints' not in self.compare.redo and self.path.exists():
            with logged_subprocess(self.compare.logger, f'reading footprints from {self.path}', timed=False):
                footprints = self.gdf = gpd.read_feather(self.path)
                return footprints
        else:
            with logged_subprocess(self.compare.logger, 'creating footprints'):
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
        if self.compare not in self.compare.__class__.data.cache:
            raise RecursionError
        data = self.compare.data
        data = data[['geometry', 'centroid']].copy()
        data['geometry'] = data['geometry'].to_crs(3857)
        data['centroid'] = data['centroid'].to_crs(3857)

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

        annoy = AnnoyIndex(2, 'euclidean')
        for i, centroid in enumerate(data['centroid']):
            annoy.add_item(i, (centroid.x, centroid.y))  # lower i, higher area
        annoy.build(10)

        for i, (g, a) in enumerate(data[['geometry', 'area']].values):
            for n in annoy.get_nns_by_item(i, 30):
                if i <= n:  # neighbor is smaller or neighbor is the same item
                    continue
                footprint: pd.Series = data.iloc[n]
                geometry: shapely.geometry.base.BaseGeometry = footprint['geometry']
                if not geometry.intersects(g):
                    continue
                # TODO: It seems that there are some cases where intersection is below 50% but
                #   it is still appropriate.
                #   Perhaps this is still necessary to prevent against odd cases.
                if geometry.intersection(g).area / a < .1:
                    continue
                footprints.iloc[i] = footprints.iloc[n]
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
        self.compare.logger(f'{len(footprints)=} from {len(data)=}')
        footprints['geometry'] = footprints['geometry'].to_crs(3857)
        footprints['centroid'] = footprints['centroid'].to_crs(3857)

        # TODO: Where is iloc applied and sorted?
        identity = self.identify(footprints)
        footprints = footprints.set_index(identity)
        self.compare.logger.debug(f'identified footprints with {identity.name=}')

        self.gdf = footprints
        return footprints

    def identify(self, gdf: GeoDataFrame) -> pd.Series:
        return pd.Series(range(len(gdf)), index=gdf.index, name='i', dtype='Int64')

    def __exit__(self, exc_type, exc_val, exc_tb):
        path = self.path
        gdf = self._gdf
        if not path.parent.exists():
            os.makedirs(path.parent)
        with logged_subprocess(self.compare.logger, f'serializing footprints to {path}', timed=False):
            gdf.to_feather(path)
        self._gdf = gdf

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
        with logged_subprocess(self.compare.logger, 'building annoy', level=logging.DEBUG):
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

        index = pd.Series(footprint(), index=data.index, name=footprints.index.name)
        names = [footprints.index.name, *data.index.names]
        data = data.set_index(index, drop=False, append=True)
        data = data.reorder_levels(names)
        data = data.sort_index(axis=0)
        return data
