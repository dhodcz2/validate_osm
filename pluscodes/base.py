from typing import Union
import posixpath

sep = posixpath.sep
import spatialpandas.geometry
from typing import Iterator

import itertools
import functools

import geopandas as gpd
import pygeos.creation

from geopandas import GeoDataFrame, GeoSeries
from pandas import IndexSlice as idx

import numpy as np
import pandas as pd
from geopandas import GeoSeries
from numpy.typing import NDArray
from pluscodes.util import *

posixpath.sep = sep # somehow, the geographic dependencies are deleting posixpath.sep

class DescriptorLoc:
    def __get__(self, instance: 'PlusCodes', owner):
        self.pluscodes = instance
        return self

    def __getitem__(self, item: np.ndarray) -> 'PlusCodes':
        pc = self.pluscodes
        if np.issubdtype(item.dtype, np.integer):
            footprints = pc.footprints.loc[item]
            heads = pc.heads.loc[idx[item, :]]
            claims = pc.claims.loc[idx[item, :, :]]

        elif np.issubdtype(item.dtype, np.string_):
            heads = pc.heads.loc[idx[:, item]]
            footprints = heads.index.get_level_values(0)
            claims = pc.claims.loc[idx[footprints, :, :]]
            footprints = pc.footprints.loc[idx[footprints]]

        elif np.issubdtype(item.dtype, np.bool_):
            heads = pc.heads.loc[item]
            footprints = pc.footprints.loc[item]
            claims = pc.claims.loc[idx[footprints.index, :, :]]

        else:
            raise TypeError(f'{item.dtype} is not supported')
        return PlusCodes(heads, footprints, claims)


#
class DescriptorIloc:
    def __get__(self, instance: 'PlusCodes', owner):
        self.pluscodes = instance
        return self

    def __getitem__(self, item) -> 'PlusCodes':
        pc = self.pluscodes
        heads = pc.heads.iloc[item]
        footprints = pc.footprints.iloc[item]
        claims = pc.claims.loc[idx[footprints.index, :, :]]
        return PlusCodes(heads, footprints, claims)


class DescriptorCx:
    def __get__(self, instance: 'PlusCodes', owner):
        self.pluscodes = instance
        return self

    def __getitem__(self, item) -> 'PlusCodes':
        pc = self.pluscodes
        footprints = pc.footprints.cx[item]
        # heads = pc.heads.loc[idx[footprints.index, :]]
        claims = pc.claims.loc[idx[footprints.index, :, :]]
        return PlusCodes(
            footprints=footprints,
            # heads=heads,
            heads=None,
            claims=claims
        )
        # return PlusCodes(heads, footprints, claims)

    def latlon(self, miny, minx, maxy, maxx) -> 'PlusCodes':
        item = (
            slice(minx, maxx),
            slice(miny, maxy)
        )
        return self.__getitem__(item)


class PlusCodes:
    loc = DescriptorLoc()
    iloc = DescriptorIloc()
    cx = DescriptorCx()

    def __init__(self, footprints: GeoSeries, heads: GeoSeries, claims: GeoSeries):
        self.footprints = footprints
        self.heads = heads
        self.claims = claims

    def __len__(self):
        return len(self.footprints)

    @functools.cached_property
    def _total_bounds(self):
        return self.footprints.total_bounds

    def __repr__(self):
        bounds = ', '.join(
            str(val)
            for val in self.footprints.total_bounds.round(2)
        )
        return f'{self.__class__.__qualname__}[{bounds}]'

    @staticmethod
    def _get_footprints(gdf: GeoDataFrame) -> GeoDataFrame:
        footprints = gdf.geometry.to_crs(epsg=4326)
        footprints = footprints.reset_index(drop=True)
        footprints.index.name = 'footprint'
        footprints: GeoDataFrame
        return footprints.geometry.to_crs(epsg=4326)

    @staticmethod
    def _get_claims(footprints: GeoSeries) -> GeoSeries:
        fw, fs, fe, fn = footprints.bounds.T.values
        lengths = get_lengths(fw, fs, fe, fn)
        longs = list(map(get_claim, fw, fs, fe, fn, lengths))
        sizes = list(map(len, longs))
        count = sum(sizes)
        longs: NDArray = np.concatenate(longs)

        # align the footprints to the claims
        iloc = np.fromiter((
            i
            for i, size in enumerate(sizes)
            for _ in range(size)
        ), dtype=np.uint64, count=count)
        # footprints = footprints.iloc[iloc]
        lengths = lengths[iloc]

        bounds = get_bounds(longs[:, 0], longs[:, 1], lengths)
        pairs: Iterator[tuple[NDArray, NDArray]] = (
            (x, y)
            for x, y in
            itertools.product((bounds[:, 0], bounds[:, 2]), (bounds[:, 1], bounds[:, 3]))
        )
        loc = np.full(count, True, dtype=bool)
        multipolygons = spatialpandas.geometry.MultiPolygonArray.from_geopandas(footprints)
        for x, y in pairs:
            points = spatialpandas.geometry.PointArray((x, y))
            intersects = points.intersects(multipolygons)
            loc &= intersects
        #     points = pygeos.creation.points(x[loc], y[loc])
        #     intersects = pygeos.intersects(footprints.geometry.values[loc], points)
        #     loc &= intersects

        # strings = get_strings(longs[loc, 0], longs[loc, 1], lengths[loc])
        strings = get_strings(longs[:, 0], longs[:, 1], lengths)
        index = pd.MultiIndex.from_arrays((
            # footprints.index[loc],
            footprints.index[iloc],
            strings
        ), names=('footprint', 'claim'))
        # bounds = bounds[loc]
        geometry = pygeos.creation.box(
            bounds[:, 0], bounds[:, 1], bounds[:, 2], bounds[:, 3]
        )
        claims = GeoSeries(
            GeoSeries(geometry, index=index, crs=4326),
            index=index,
            crs=4326
        )
        return claims

    # @staticmethod
    # def _get_heads(claims: GeoSeries) -> GeoSeries:
    #     groups = claims.groupby(level='claim').groups
    #     loc = [
    #         group[0]
    #         for group in groups.values()
    #     ]
    #     heads = claims.loc[loc]
    #     claims.drop(index=loc, inplace=True)
    #     return heads
    #
    @classmethod
    def from_gdf(cls, gdf: GeoSeries | GeoDataFrame) -> 'PlusCodes':
        footprints = cls._get_footprints(gdf)
        claims = cls._get_claims(footprints)
        # heads = cls._get_heads(claims)
        return cls(
            footprints=footprints,
            heads=None,
            claims=claims,
        )

    @classmethod
    def from_file(cls, filepath: str) -> 'PlusCodes':
        extension = filepath.rpartition('.')[-1]
        if extension == 'feather':
            footprints = gpd.read_feather(filepath)
        elif extension == 'parquet':
            footprints = gpd.read_parquet(filepath)
        else:
            footprints = gpd.read_file(filepath)
        return cls.from_gdf(footprints)

    def xs(self, key: Union[int, str], level) -> 'PlusCodes':
        if level == 'footprint':
            footprints = self.footprints.xs(key)
            heads = self.heads.loc[idx[footprints.index, :]]
            claims = self.claims.loc[idx[footprints.index, :, :]]
        elif level == 'head':
            heads = self.heads.xs(key, level='head')
            footprint = heads.index.get_level_values(0)
            claims = self.claims.loc[idx[footprint, :, :]]
            footprints = self.footprints.loc[idx[footprint]]
        elif level == 'claim':
            claims = self.claims.xs(key, level='claim')
            footprint = claims.index.get_level_values(0)
            heads = self.heads.loc[idx[footprint.index, :]]
            footprints = self.footprints.loc[idx[footprint]]
        else:
            raise ValueError(f'{level} is not supported')
        return PlusCodes(footprints, heads, claims)

    def explore(self, **kwargs) -> None:
        centroid = self.footprints.iloc[0].centroid
        import folium
        map = folium.Map(
            location=(centroid.y, centroid.x),
            zoom_start=16,
        )
        footprints: GeoSeries = self.footprints
        footprints: GeoDataFrame = GeoDataFrame({
            # 'footprint': footprints.index.get_level_values('footprint'),
        }, geometry=footprints.geometry, crs=4326, index=footprints.index)

        heads: GeoSeries = self.heads
        if heads is not None:
            heads: GeoDataFrame = GeoDataFrame({
                'head': heads.index.get_level_values('head'),
            }, geometry=heads.geometry, crs=4326, index=heads.index)

        claims: GeoSeries = self.claims
        claims: GeoDataFrame = GeoDataFrame({
            'claim': claims.index.get_level_values('claim'),
        }, geometry=claims.geometry, crs=4326, index=claims.index)
        # loc = claims.index.get_level_values('claim') != claims.index.get_level_values('head')
        # claims = claims.loc[loc]

        footprints.explore(
            m=map,
            color='black',
            style_kwds=dict(
                fill=False,
            )
        )
        if heads is not None:
            heads.explore(
                m=map,
                color='blue',
            )
        claims.explore(
            m=map,
            color='red',
        )
        return map


if __name__ == '__main__':
    pc = PlusCodes.from_file('/home/arstneio/Downloads/chicago.feather')
    pc.explore()
    print()
