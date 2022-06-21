import functools
import folium

import geopandas as gpd
import pygeos.creation

from geopandas import GeoDataFrame, GeoSeries
from pandas import IndexSlice as idx

import numpy as np
import pandas as pd
from geopandas import GeoSeries
from numpy.typing import NDArray
from pluscodes.util import *


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
        heads = pc.heads.loc[idx[footprints.index, :]]
        claims = pc.claims.loc[idx[footprints.index, :, :]]
        return PlusCodes(
            footprints=footprints,
            heads=heads,
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

    @classmethod
    def from_footprints(cls, footprints: GeoSeries | GeoDataFrame):
        footprints = footprints.geometry.to_crs(epsg=4326)
        footprints = footprints.reset_index(drop=True)
        footprints.index.name = 'footprint'
        footprints: GeoDataFrame

        fw, fs, fe, fn = footprints.bounds.T.values

        # heads
        lengths = get_lengths(fw, fs, fe, fn)
        points = footprints.representative_point()
        fx = points.x.values
        fy = points.y.values
        heads = get_strings(x=fx, y=fy, lengths=lengths)
        index = pd.MultiIndex.from_arrays((
            np.arange(len(footprints)),
            heads,
        ), names=('footprint', 'head'))
        bounds = get_bounds(fx, fy, lengths)
        geometry = pygeos.creation.box(
            bounds[:, 0], bounds[:, 1], bounds[:, 2], bounds[:, 3]
        )
        head = GeoSeries(geometry, index=index, crs=4326)

        # claims
        list_claims: list[NDArray[np.uint64]] = [
            get_claim(w, s, e, n, l)
            for w, s, e, n, l in zip(fw, fs, fe, fn, lengths)
        ]
        longs = np.concatenate(list_claims)
        lx = longs[:, 0]
        ly = longs[:, 1]
        count = sum(map(len, list_claims))
        iloc = np.fromiter((
            i
            for i, list_ in enumerate(list_claims)
            for _ in range(len(list_))
        ), dtype=np.uint64, count=count)
        lengths = lengths[iloc]
        claims = get_strings(x=lx, y=ly, lengths=lengths)
        index = pd.MultiIndex.from_arrays((
            iloc, heads[iloc], claims
        ), names=('footprint', 'head', 'claim'))
        bounds = get_bounds(lx, ly, lengths)
        geometry = pygeos.creation.box(
            bounds[:, 0], bounds[:, 1], bounds[:, 2], bounds[:, 3]
        )
        claims = GeoSeries(geometry, index=index, crs=4326)
        # TODO: in the morning, graphically investigate and implement pytest

        return cls(
            footprints=footprints,
            heads=head,
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
        return cls.from_footprints(footprints)

    def xs(self, key: int | str, level) -> 'PlusCodes':
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
        map = folium.Map(location=(centroid.y, centroid.x))
        footprints: GeoSeries = self.footprints
        footprints: GeoDataFrame = GeoDataFrame({
            # 'footprint': footprints.index.get_level_values('footprint'),
        }, geometry=footprints.geometry, crs=4326, index=footprints.index)

        heads: GeoSeries = self.heads
        heads: GeoDataFrame = GeoDataFrame({
            'head': heads.index.get_level_values('head'),
        }, geometry=heads.geometry, crs=4326, index=heads.index)

        claims: GeoSeries = self.claims
        claims: GeoDataFrame = GeoDataFrame({
            'claim': claims.index.get_level_values('claim'),
        }, geometry=claims.geometry, crs=4326, index=claims.index)
        loc = claims.index.get_level_values('claim') != claims.index.get_level_values('head')
        claims = claims.loc[loc]

        footprints.explore(
            m=map,
            color='black',
            style_kwds=dict(
                fill=False,
            )
        )
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
    print()
