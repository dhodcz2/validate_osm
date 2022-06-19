import itertools

import pandas as pd
import pygeos.creation
from numpy.typing import NDArray
import functools
import numpy as np
import math
from typing import NamedTuple
from geopandas import GeoDataFrame, GeoSeries
from pandas import Series, DataFrame

Bounds = NamedTuple('Bounds', (('s', np.ndarray), ('w', np.ndarray), ('n', np.ndarray), ('e', np.ndarray)))
from pluscodes.util import (
    get_claim,
    get_strings,
    get_lengths,
    get_string,
    get_bound,
    get_bounds,
)


class DescriptorLoc:
    def __get__(self, instance: 'PlusCodes', owner):
        self.pluscodes = instance
        return self

    def __getitem__(self, item) -> 'PlusCodes':
        heads = self.pluscodes.heads.loc[item]
        tails = self.pluscodes.tails.loc[item]
        return PlusCodes(heads, tails)


class DescriptorIloc:
    def __get__(self, instance: 'PlusCodes', owner):
        self.pluscodes = instance
        return self

    def __getitem__(self, item) -> 'PlusCodes':
        heads = self.pluscodes.heads.iloc[item]
        claims = self.pluscodes.claims.iloc[item]
        return PlusCodes(heads, claims)


class DescriptorCx:
    def __get__(self, instance: 'PlusCodes', owner):
        self.pluscodes = instance
        return self

    def __getitem__(self, item) -> 'PlusCodes':
        heads = self.pluscodes.heads.cx[item]
        claims = self.pluscodes.claims.cx[item]
        return PlusCodes(heads, claims)


class PlusCodes:
    # geometry = DescriptorGeometry
    loc = DescriptorLoc()
    iloc = DescriptorIloc()
    cx = DescriptorCx()

    def __init__(self, heads: GeoSeries, claims: GeoSeries):
        self.heads = heads
        self.claims = claims

    @classmethod
    def from_bounds(cls, fw, fs, fe, fn):
        lengths = get_lengths(fw, fs, fe, fn)
        x = (fe + fw) / 2
        y = (fn + fs) / 2
        heads = get_strings(x=x, y=y, lengths=lengths, )
        index = pd.Index(heads, name='head')
        geometry = get_bounds(x, y, lengths)
        head = GeoSeries(geometry, index=index, crs=4326)

        list_longs: list[NDArray[np.uint64]] = [
            get_claim(w, s, e, n, l)
            for w, s, e, n, l in zip(fw, fs, fe, fn, lengths)
        ]
        longs = np.concatenate(list_longs)
        x = longs[:, 0]
        y = longs[:, 1]
        claims = get_strings(x=x, y=y, lengths=lengths, )
        index = pd.MultiIndex.from_tuples((
            (head, claim)
            for head, arr, claim in zip(heads, list_longs, claims)
            for _ in range(len(arr))
        ))
        geometry = get_bounds(x, y, lengths)
        claims = GeoSeries(geometry, index=index, crs=4326)

        return cls(
            heads=head,
            claims=claims,
        )
