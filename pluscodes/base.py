from typing import NamedTuple
from numpy.typing import NDArray

import pygeos

SEPARATOR_ = '+'
SEPARATOR_POSITION_ = 8
PADDING_CHARACTER_ = '0'
CODE_ALPHABET_ = '23456789CFGHJMPQRVWX'
ENCODING_BASE_ = len(CODE_ALPHABET_)
LATITUDE_MAX_ = 90
LONGITUDE_MAX_ = 180
MAX_DIGIT_COUNT_ = 15
PAIR_CODE_LENGTH_ = 10
PAIR_FIRST_PLACE_VALUE_ = ENCODING_BASE_ ** (PAIR_CODE_LENGTH_ / 2 - 1)
PAIR_PRECISION_ = ENCODING_BASE_ ** 3
PAIR_RESOLUTIONS_ = [20.0, 1.0, .05, .0025, .000125]
GRID_CODE_LENGTH_ = MAX_DIGIT_COUNT_ - PAIR_CODE_LENGTH_
GRID_COLUMNS_ = 4
GRID_ROWS_ = 5
GRID_LAT_FIRST_PLACE_VALUE_ = GRID_ROWS_ ** (GRID_CODE_LENGTH_ - 1)
GRID_LNG_FIRST_PLACE_VALUE_ = GRID_COLUMNS_ ** (GRID_CODE_LENGTH_ - 1)
FINAL_LAT_PRECISION_ = PAIR_PRECISION_ * GRID_ROWS_ ** (MAX_DIGIT_COUNT_ - PAIR_CODE_LENGTH_)
FINAL_LNG_PRECISION_ = PAIR_PRECISION_ * GRID_COLUMNS_ ** (MAX_DIGIT_COUNT_ - PAIR_CODE_LENGTH_)
MIN_TRIMMABLE_CODE_LEN_ = 6
GRID_SIZE_DEGREES_ = 0.000125

import functools

import numpy as np
import math

Bounds = NamedTuple('Bounds', (('s', np.ndarray), ('w', np.ndarray), ('n', np.ndarray), ('e', np.ndarray)))


def pcode_join(

):
    """
        Are there any other pcodes, that this particular geometry may satisfy?
        So, all currently established pcodes, which ones can this also belong to?
    :return:
    """
    ...


"""
    Determine code length from building size
        floored; so a building's plus code area should be smaller than its actual area
"""
"""
    Generate gdf.to_crs(4326).total_bounds
        from those bounds, generate UBIDS
        from other ubids, determine which smaller UBIDs are within the range of that larger UBID
"""


class PlusCodes:
    def __init__(self, iy: np.ndarray, ix: np.ndarray, code_lengths: np.ndarray):
        self.iy = iy
        self.ix = ix
        self.code_lengths = code_lengths
        self.alphabet = np.fromiter(CODE_ALPHABET_, (np.unicode, 1))

    @classmethod
    def _from_points(cls, gy: np.ndarray, gx: np.ndarray, code_lengths: np.ndarray):
        if gy.dtype != np.float64:
            raise TypeError(f'{np.float64}!={gy.dtype=}')
        if gx.dtype != np.float64:
            raise TypeError(f'{np.float64}!={gx.dtype=}')
        rows = len(code_lengths)

        where = np.where(np.logical_or(
            code_lengths < 2,
            np.logical_and(
                code_lengths < PAIR_CODE_LENGTH_,
                code_lengths % 2 == 1
            )
        ))
        if len(where):
            raise ValueError(f'invalid open location code lengths at {where}')

        code_lengths[code_lengths > MAX_DIGIT_COUNT_] = MAX_DIGIT_COUNT_

        # -180 < lons <= 180
        first = gx < 0
        gx[first] %= -360
        gx[~first] %= 360
        gx[gx == -180] = 180

        # lats >= -90
        gy[gy < -90] = -90

        # lats < 90
        first = gy >= 90
        ceil_lats = code_lengths[first]
        second = ceil_lats < 10
        ceil_lats[second] = np.power(20, np.floor((ceil_lats[second] / -2) + 2))
        ceil_lats[~second] = math.pow(20, -3) / np.power(GRID_ROWS_, ceil_lats[~second] - 10)
        gy[first] = ceil_lats

        iy = (gy + LATITUDE_MAX_) * FINAL_LAT_PRECISION_
        ix = (gx + LONGITUDE_MAX_) * FINAL_LNG_PRECISION_
        iy = np.ndarray.astype(iy, np.uint64)
        ix = np.ndarray.astype(ix, np.uint64)

        return cls(iy, ix, code_lengths)

    @classmethod
    def from_points(
            cls,
            gy: NDArray[np.float64] | float,
            gx: NDArray[np.float64] | float,
            code_lengths: NDArray[np.uint] | int
    ):
        """
        Given centroids and specified code length, create plus codes.
        :param gy:  geographic latitude
        :param gx:  geographic longitude
        :param code_lengths: length of point code
        :return:
        """
        if isinstance(gy, float):
            gy = np.array(gy, dtype=np.float64)
        if isinstance(gx, float):
            gx = np.array(gx, dtype=np.float64)
        if isinstance(code_lengths, float):
            code_lengths = np.array(code_lengths, dtype=np.float64)
        return cls._from_points(gy, gx, code_lengths)

    @classmethod
    def from_bounds(cls, gw: np.ndarray, gs: np.ndarray, ge: np.ndarray, gn: np.ndarray):
        gy = (gn + gs) / 2
        gx = (ge + gw) / 2
        code_lengths = NotImplemented
        # TODO: determine code lengths from the bboxes
        return cls._from_points(gy, gx, code_lengths)

    @classmethod
    def from_codes(cls):
        ...

    @functools.cached_property
    def pairs(self) -> np.ndarray:
        ix = self.ix
        iy = self.iy
        rows = len(ix)
        alphabet = self.alphabet
        pairs = np.ndarray((rows, PAIR_CODE_LENGTH_), dtype='U')
        for column, precision in enumerate(
                ENCODING_BASE_ ** n for n in range(PAIR_CODE_LENGTH_ // 2)[::-1]
        ):
            pairs[:, column] = alphabet[ix // precision % ENCODING_BASE_]
            pairs[:, column + 1] = alphabet[iy // precision % ENCODING_BASE_]

        return pairs

    @functools.cached_property
    def grids(self) -> np.ndarray:
        ix = self.ix
        iy = self.iy
        rows = len(ix)
        alphabet = self.alphabet
        grids = np.ndarray((rows, GRID_CODE_LENGTH_), dtype='U')
        for column, (precision_lat, precision_lon) in enumerate(zip(
                (GRID_ROWS_ ** n for n in range(GRID_CODE_LENGTH_)[::-1]),
                (GRID_COLUMNS_ ** n for n in range(GRID_CODE_LENGTH_)[::-1]),
        )):
            mod = iy // precision_lat % GRID_ROWS_ * GRID_COLUMNS_
            mod += ix // precision_lon % GRID_COLUMNS_
            grids[:, column] = alphabet[mod]

        return grids

    @functools.cached_property
    def bounds(self) -> np.ndarray:
        ix = self.ix
        iy = self.iy
        rows = len(ix)
        # Pair section is simple
        fx = (ix / FINAL_LNG_PRECISION_) - LONGITUDE_MAX_
        fy = (iy / FINAL_LAT_PRECISION_) - LATITUDE_MAX_

        gw = fx
        gs = fy
        ge = fx
        gn = fy

        # Now determine the resolution of the grids,
        # remove the pairs contribution
        pairs: int = (ENCODING_BASE_ ** (PAIR_CODE_LENGTH_ // 2))
        # TODO: //= or /=?
        ix //= pairs
        iy //= pairs
        px = np.full(rows, 0, dtype=np.float64)
        py = np.full(rows, 0, dtype=np.float64)

        for precision in (GRID_ROWS_ ** n for n in range(GRID_CODE_LENGTH_)):
            py[iy // precision > 0] = precision
        for precision in (GRID_COLUMNS_ ** n for n in range(GRID_CODE_LENGTH_)):
            px[ix // precision > 0] = precision

        ge += px / FINAL_LAT_PRECISION_
        gn += py / FINAL_LNG_PRECISION_

        return np.concatenate((gw, gs, ge, gn))

    @functools.cached_property
    def codes(self) -> np.ndarray:
        pairs = self.pairs
        grids = self.grids
        left = pairs[:, :SEPARATOR_POSITION_]
        right = np.char.join((pairs[:, SEPARATOR_POSITION_:], grids), sep='')
        codes = np.char.join((left, right), sep=SEPARATOR_)
        return codes

    @functools.cached_property
    def geometries(self) -> np.ndarray:
        import pygeos.creation
        return pygeos.creation.box(*self.mercator)

    def __len__(self):
        return len(self.gs)

    def __and__(self, other: 'PlusCodes'):
        ...

    def __or__(self, other: 'PlusCodes'):
        ...

    def contains(self, item: 'PlusCodes') -> np.ndarray:
        return np.logical_and(
            item.integers.w >= self.integers.w,
            item.integers.s >= self.integers.s,
            item.integers.e <= self.integers.e,
            item.integers.n <= self.integers.n,
        )

    @functools.cached_property
    def bytes(self) -> Bounds:
        raise NotImplementedError

    @functools.cached_property
    def integers(self) -> Bounds:
        raise NotImplementedError

    @functools.cached_property
    def strings(self) -> np.ndarray:
        raise NotImplementedError

    def __repr__(self):
        raise NotImplementedError


if __name__ == '__main__':
    pcodes = PlusCodes.from_points(47.365590, 8.524997, 11)
