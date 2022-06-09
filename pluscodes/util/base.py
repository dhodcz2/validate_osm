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
    encode_string,
    encode_digits,
    decode_digits,
)


# class _PlusCodes:
#     def __init__(self, iy: np.ndarray, ix: np.ndarray, code_lengths: np.ndarray):
#         self.iy = iy
#         self.ix = ix
#         self.code_lengths = code_lengths
#
#     @classmethod
#     def _from_points(cls, gy: np.ndarray, gx: np.ndarray, code_lengths: np.ndarray):
#         if gy.dtype != np.float64:
#             raise TypeError(f'{np.float64}!={gy.dtype=}')
#         if gx.dtype != np.float64:
#             raise TypeError(f'{np.float64}!={gx.dtype=}')
#         rows = len(code_lengths)
#
#         where, = np.where(np.logical_or(
#             code_lengths < 2,
#             np.logical_and(
#                 code_lengths < PAIR_CODE_LENGTH_,
#                 code_lengths % 2 == 1
#             )
#         ))
#         if len(where):
#             raise ValueError(f'invalid open location code lengths at {where}')
#
#         code_lengths[code_lengths > MAX_DIGIT_COUNT_] = MAX_DIGIT_COUNT_
#
#         # -180 < lons <= 180
#         first = gx < 0
#         gx[first] %= -360
#         gx[~first] %= 360
#         gx[gx == -180] = 180
#
#         # lats >= -90
#         gy[gy < -90] = -90
#
#         # lats < 90
#         first = gy >= 90
#         ceil_lats = code_lengths[first]
#         second = ceil_lats < 10
#         ceil_lats[second] = np.power(20, np.floor((ceil_lats[second] / -2) + 2))
#         ceil_lats[~second] = math.pow(20, -3) / np.power(GRID_ROWS_, ceil_lats[~second] - 10)
#         gy[first] = ceil_lats
#
#         iy = (gy + LATITUDE_MAX_) * FINAL_LAT_PRECISION_
#         ix = (gx + LONGITUDE_MAX_) * FINAL_LNG_PRECISION_
#         iy = np.ndarray.astype(iy, np.uint64)
#         ix = np.ndarray.astype(ix, np.uint64)
#
#         return cls(iy, ix, code_lengths)
#
#     @classmethod
#     def _from_region(
#             cls,
#             gw: float,
#             gs: float,
#             ge: float,
#             gn: float,
#             code_lengths: int,
#     ) -> 'PlusCodes':
#         raise NotImplementedError()
#
#
#
#
# class PlusCodes(_PlusCodes):
#     @classmethod
#     def from_points(
#             cls,
#             gy: NDArray[np.float64] | float,
#             gx: NDArray[np.float64] | float,
#             code_lengths: NDArray[np.uint] | int,
#     ):
#         """
#         Given centroids and specified code length, create plus codes.
#         :param gy:  geographic latitude
#         :param gx:  geographic longitude
#         :param code_lengths: length of point code
#         :return:
#         """
#         if isinstance(gy, float):
#             gy = np.array(gy, dtype=np.float64, ndmin=1)
#         if isinstance(gx, float):
#             gx = np.array(gx, dtype=np.float64, ndmin=1)
#         if isinstance(code_lengths, int):
#             code_lengths = np.array(code_lengths, dtype=np.uint, ndmin=1)
#         return cls._from_points(gy, gx, code_lengths)
#
#     @classmethod
#     def from_region(
#             cls,
#             gw: float,
#             gs: float,
#             ge: float,
#             gn: float,
#             code_lengths: int,
#     ) -> 'PlusCodes':
#         return cls._from_region(gw, gs, ge, gn, code_lengths)
#
#     @classmethod
#     def from_bounds(
#             cls,
#             gw: NDArray[np.float64] | float,
#             gs: NDArray[np.float64] | float,
#             ge: NDArray[np.float64] | float,
#             gn: NDArray[np.float64] | float,
#             contains: bool = False,
#     ) -> 'PlusCodes':
#         """
#
#         :param gw: geographic western bound
#         :param gs: geographic southern bound
#         :param ge: geographic eastern bound
#         :param gn: geoggraphic northern bound
#         :param contains:
#             True:
#                 the code represents a tile that contains the bounds
#             False:
#                 the code represents a tile that is contained by the bounds
#         :return:
#         """
#         if isinstance(gw, float):
#             gw = np.array(gw, dtype=np.float64, ndmin=1)
#             gs = np.array(gs, dtype=np.float64, ndmin=1)
#             ge = np.array(ge, dtype=np.float64, ndmin=1)
#             gn = np.array(gn, dtype=np.float64, ndmin=1)
#         return cls._from_bounds(gw, gs, ge, gn, contains)
#
#

class PlusCodes:
    def __init__(self, iy: np.ndarray, ix: np.ndarray, code_lengths: np.ndarray):
        self.iy = iy
        self.ix = ix
        self.code_lengths = code_lengths

    @functools.cached_property
    def index(self) -> NDArray[np.uint64]:
        return encode_string(self.iy, self.ix, self.code_lengths)

    @functools.cached_property
    def geometry(self) -> NDArray[np.uint64]:
        '''turn self.bounds into an array of boxes'''
        return pygeos.creation.box(*self.bounds)

    @functools.cached_property
    def bounds(self) -> NDArray[np.float64]:
        ...

    @functools.cached_property
    def gdf(self) -> GeoDataFrame:
        return GeoDataFrame(geometry=self.geometry, crs=4326, index=self.index)

