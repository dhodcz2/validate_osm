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
        return PlusCodes(heads, footprints, claims)


class PlusCodes:
    loc = DescriptorLoc()
    iloc = DescriptorIloc()
    cx = DescriptorCx()

    def __init__(self, footprints: GeoSeries, heads: GeoSeries, claims: GeoSeries):
        self.footprints = footprints
        self.heads = heads
        self.claims = claims

    @classmethod
    def from_footprints(cls, footprints: GeoSeries | GeoDataFrame):
        footprints = footprints.geometry.to_crs(epsg=4326)
        footprints = footprints.reset_index(drop=True)
        fw, fs, fe, fn = footprints.bounds.T.values

        # heads
        lengths = get_lengths(fw, fs, fe, fn)
        fx = (fe + fw) / 2
        fy = (fn + fs) / 2
        heads = get_strings(x=fx, y=fy, lengths=lengths, )
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

    def explore(self, *args, **kwargs) -> None:
        ...



if __name__ == '__main__':
    pc = PlusCodes.from_file('/home/arstneio/Downloads/chicago.feather')
    print()

