import functools
import itertools
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


class CallableFootprint:
    # def __init__(self, data: GeoDataFrame, path: Path, ignore_file: bool = False):
    #     if not ignore_file and path.exists():
    #         with logged_subprocess(logger, f'reading footprints from {path}'):
    #             self.footprints = gpd.read_feather(path)
    #     else:
    #         with logged_subprocess(logger, 'building footprints'):
    #             self.footprints = self._footprints(data)
    #         logger.info(f'footprints built {len(self.footprints)=} from {len(data)=}')
    #
    #         if not path.parent.exists():
    #             os.makedirs(path)
    #         with logged_subprocess(logger, f'serializing footprints to {path}'):
    #             self.footprints.to_feather(path)
    def __init__(self, compare: object):
        from validate_osm.compare.compare import Compare
        compare: Compare
        self.compare = compare
        path = compare.directory / 'footprint.feather'
        if not compare.ignore_file and path.exists():
            with logged_subprocess(compare.logger, f'reading footprints from {path}'):
                self.footprints = gpd.read_feather(path)
        else:
            data = compare.data
            with logged_subprocess(compare.logger, 'building footprints'):
                self.footprints = self._footprints(data)
                compare.logger.info(f'{len(self.footprints)=} from {len(data)=}')

            if not path.parent.exists():
                os.makedirs(path.parent)
            with logged_subprocess(compare.logger, f'serializing footprints to {path}'):
                self.footprints.to_feather(path)

    def _footprints(self, data: GeoDataFrame) -> GeoDataFrame:
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
        identity = self.identity(footprints)
        self.compare.logger.debug(f'identified footprints with {identity.name=}')
        footprints = footprints.set_index(identity)

        footprints['geometry'] = footprints['geometry'].to_crs(3857)
        footprints['centroid'] = footprints['centroid'].to_crs(3857)

        return footprints

    def identity(self, gdf: GeoDataFrame) -> pd.Series:
        return pd.Series(range(len(gdf)), index=gdf.index, name='i', dtype='Int64')

    @functools.cached_property
    def _annoy(self) -> AnnoyIndex:
        annoy = AnnoyIndex(2, 'euclidean')
        for i, centroid in enumerate(self.footprints['centroid']):
            annoy.add_item(i, (centroid.x, centroid.y))
        annoy.build(10)
        return annoy

    def __call__(self, data: GeoDataFrame) -> GeoDataFrame:
        data['geometry'] = data['geometry'].to_crs(3857)
        data['centroid'] = data['centroid'].to_crs(3857)
        data['area'] = data['geometry'].area
        footprints = self.footprints

        def footprint() -> Generator[Hashable, None, None]:
            # TODO: ValueError: Length of values (74709) does not match length of index (74999)
            #   Therefore, there has been some missed matches (with n=5).
            for i, (c, a, g) in enumerate(data[['centroid', 'area', 'geometry']].values):
                for n in self._annoy.get_nns_by_vector((c.x, c.y), 10):
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
        data = data.sort_index()
        return data
