import numpy as np
import math
from pluscodes.util import head, pad

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


def _ceil_lats(code_lengths: np.ndarray):
    return 90 - np.where(
        code_lengths < 10,
        np.power(20, np.floor((code_lengths / -2) + 2)),
        math.pow(20, -3) / np.power(GRID_ROWS_, code_lengths - 10)
    )


# TODO


def encode(
        lats: np.ndarray,
        lons: np.ndarray,
        code_lengths: np.ndarray,
) -> np.ndarray:
    if lats.dtype != np.float64:
        raise TypeError(f'{np.float64}!={lats.dtype=}')
    if lons.dtype != np.float64:
        raise TypeError(f'{np.float64}!={lons.dtype=}')
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
    first = lons < 0
    lons[first] %= -360
    lons[~first] %= 360
    lons[lons == -180] = 180

    # lats >= -90
    lats[lats < -90] = -90

    # lats < 90
    first = lats >= 90
    ceil_lats = code_lengths[first]
    second = ceil_lats < 10
    ceil_lats[second] = np.power(20, np.floor((ceil_lats[second] / -2) + 2))
    ceil_lats[~second] = math.pow(20, -3) / np.power(GRID_ROWS_, ceil_lats[~second] - 10)
    lats[first] = ceil_lats

    lats = (lats + LATITUDE_MAX_) * FINAL_LAT_PRECISION_
    lons = (lons + LONGITUDE_MAX_) * FINAL_LAT_PRECISION_
    lats = lats.astype(np.uint64)
    lons = lons.astype(np.uint64)

    alphabet = np.array(CODE_ALPHABET_)

    # Compute the grid part of the code if necessary
    grids = np.ndarray((rows, GRID_CODE_LENGTH_), dtype='U')
    for i, (precision_lat, precision_lon) in enumerate(zip(
            (GRID_ROWS_ ** n for n in range(GRID_CODE_LENGTH_)[::-1]),
            (GRID_COLUMNS_ ** n for n in range(GRID_CODE_LENGTH_)[::-1]),
    )):
        grids[:, i] = alphabet[
            (lats // precision_lat % GRID_ROWS_ * GRID_COLUMNS_)
            + (lons // precision_lon % GRID_COLUMNS_)
        ]


    # Compute the pairs
    pairs = np.ndarray((rows, PAIR_CODE_LENGTH_), dtype='U')
    for i, precision in enumerate(
            ENCODING_BASE_ ** n for n in range(PAIR_CODE_LENGTH_ // 2)[::-1]
    ):
        pairs[:, i] = alphabet[lons // precision % ENCODING_BASE_]
        pairs[:, i + 1] = alphabet[lats // precision % ENCODING_BASE_]

    # for i, precision in enumerate(ENCODING_BASE_**n for n in range())

    # grid_code = np.where(code_lengths > PAIR_CODE_LENGTH_)
    # not_grid_code = np.where(code_lengths < PAIR_CODE_LENGTH_)
    #
    # lats[grid_code], lons[grid_code], codes[grid_code] = _grid_code(
    #     lats[grid_code], lons[grid_code], codes[grid_code]
    # )
    #
    # lats[not_grid_code] //= pow(GRID_ROWS_, GRID_CODE_LENGTH_)
    # lons[not_grid_code] //= pow(GRID_COLUMNS_, GRID_CODE_LENGTH_)
    #
    # codes = np.char.join(codes[:SEPARATOR_POSITION_], SEPARATOR_, codes[SEPARATOR_POSITION_:])
    #
    # codes = np.where(
    #     code_lengths > SEPARATOR_POSITION_,
    #     head(codes, code_lengths),
    #     codes
    # )
    #
    # codes = pad(codes, code_lengths)
    # return codes
