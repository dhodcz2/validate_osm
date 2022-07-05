import pandas as pd
import pygeos.creation
from numpy.typing import NDArray
from typing import Union

import spatialpandas.geometry.base
from libc.stdlib cimport malloc, free
# from cpython cimport Py_buffer
from libcpp cimport vector
import numpy as np
cimport numpy as np
import geopandas as gpd
import shapely.geometry
import spatialpandas.geometry
import shapely.geometry
cimport util.cfuncs as cfuncs

np.string_t = np.dtype('S%d' % (MAX_DIGITS + 1))
ctypedef np.uint8_t UINT8
ctypedef np.uint16_t UINT16
ctypedef np.float64_t F64
ctypedef np.uint32_t UINT32
ctypedef np.uint64_t UINT64
ctypedef np.int64_t INT64
ctypedef np.uint_t UINT
ctypedef np.int_t INT
ctypedef np.uint8_t BOOL

cdef extern from '<util/globals.h>':
    char SEP
    unsigned int SEP_POS
    char* ALPHABET
    unsigned int BASE
    char PAD
    unsigned int MAX_LAT
    unsigned int MAX_LON
    unsigned int MAX_DIGITS
    unsigned int PAIR_LENGTH
    unsigned int PAIR_PRECISION_FIRST_VALUE
    unsigned long PAIR_PRECISION
    unsigned int GRID_LENGTH
    unsigned int GRID_COLUMNS
    unsigned int GRID_ROWS
    unsigned int GRID_LAT_FIRST_PLACE_VALUE
    unsigned int GRID_LON_FIRST_PLACE_VALUE
    unsigned long FINAL_LAT_PRECISION
    unsigned long FINAL_LON_PRECISION
    unsigned int MIN_TRIMMABLE_CODE_LEN
    double GRID_SIZE_DEGREES

    unsigned long GRID_LAT_RESOLUTION
    unsigned long GRID_LON_RESOLUTION


cdef class ShapeClaim:
    points: spatialpandas.geometry.PointArray
    footprint: spatialpandas.geometry.base.Geometry
    cdef unsigned char length

    # cdef np.ndarray[UINT8, ndim=1, cast=True] visited
    # cdef np.ndarray[UINT8, ndim=1, cast=True] contained

    cdef np.uint8_t[:] visited
    cdef np.uint8_t[:] contained

    cdef Py_ssize_t claimrows
    cdef Py_ssize_t claimcols
    cdef Py_ssize_t pointrows
    cdef Py_ssize_t pointcols
    cdef Py_ssize_t size

    cdef Py_ssize_t shape[2]
    cdef Py_ssize_t strides[2]

    cdef unsigned long[:, :] longs


    def __cinit__(
            self,
            footprint: shapely.geometry.base.BaseGeometry,
            unsigned char length,
            unsigned long lw,
            unsigned long ls,
            unsigned long le,
            unsigned long ln,
    ):
        cdef Py_ssize_t r, c, i

        cdef unsigned long xstep = pow(GRID_COLUMNS, MAX_DIGITS - length)
        cdef unsigned long ystep = pow(GRID_ROWS, MAX_DIGITS - length)

        self.claimcols = le // xstep - lw // xstep
        self.claimrows = ln // ystep - ls // ystep
        self.pointrows = self.claimrows + 1
        self.pointcols = self.claimcols + 1

        # (left, right], (bottom, top]
        cdef np.ndarray[UINT64, ndim=1] dx = np.arange(xstep, self.pointcols * xstep + 1, xstep, dtype=np.uint64)
        cdef np.ndarray[UINT64, ndim=1] dy = np.arange(ystep, self.pointrows * ystep + 1, ystep, dtype=np.uint64)

        cdef longs = np.ndarray(shape=(self.pointrows * self.pointcols, 2), dtype=np.uint64)
        cdef unsigned long[:,:] longv = longs

        for r in range(self.pointrows):
            for c in range(self.pointcols):
                longv[<Py_ssize_t> (r*self.pointcols+c), 0] = lw + dx[c]
                longv[<Py_ssize_t> (r*self.pointcols+c), 1] = ls + dy[r]

        cdef np.ndarray[F64, ndim=2] coords = np.ndarray(shape=(self.pointcols*self.pointrows, 2), dtype=np.float64)
        cdef double[:, :] floatv = coords
        cdef unsigned long final_lat_precision = PAIR_PRECISION * pow(GRID_ROWS, length - PAIR_LENGTH)
        cdef unsigned long final_lon_precision = PAIR_PRECISION * pow(GRID_COLUMNS, length - PAIR_LENGTH)
        cdef unsigned long trim_lats = pow(<unsigned long> GRID_ROWS, MAX_DIGITS-length)
        cdef unsigned long trim_lons = pow(<unsigned long> GRID_COLUMNS, MAX_DIGITS-length)

        for r in range(self.pointrows):
            for c in range(self.pointcols):
                i = r*self.pointcols+c
                floatv[i, 0] = <double>(longv[i,0] // trim_lons) / final_lon_precision - MAX_LON
                floatv[i, 1] = <double>(longv[i,1] // trim_lats) / final_lat_precision - MAX_LAT

        self.length = length
        self.visited = np.full(shape=(self.pointrows*self.pointcols), fill_value=0, dtype=np.uint8)
        self.contained = np.full(shape=(self.pointrows*self.pointcols), fill_value=0, dtype=np.uint8)
        self.points = spatialpandas.geometry.PointArray((coords[:, 0], coords[:, 1]))
        self.longs = longv
        self.size = 0

    def __init__(
            self,
            footprint: spatialpandas.geometry.base.Geometry,
            unsigned char length,
            unsigned long lw,
            unsigned long ls,
            unsigned long le,
            unsigned long ln,
    ):
        cdef Py_ssize_t r, c, n, i, sw, se, nw, ne
        self.kickoff(0, 0, self.claimcols, self.claimrows)

        n = 0
        for r in range(self.claimrows):
            for c in range(self.claimcols):
                sw = r * self.pointcols + c
                se = r * self.pointcols + c + 1
                nw = (r + 1) * self.pointcols + c
                ne = (r + 1) * self.pointcols + c + 1
                if  self.contained[sw] and self.contained[se] and self.contained[nw] and self.contained[ne] :
                    self.longs[n, 0] = self.longs[sw, 0]
                    self.longs[n, 1] = self.longs[sw, 1]
                    n += 1

    def __len__(self):
        return self.size

    cdef inline void kickoff(self, Py_ssize_t w, Py_ssize_t s,  Py_ssize_t e, Py_ssize_t n):
        cdef Py_ssize_t sw, se, nw, ne, midx, midy, r, c
        sw = s * self.pointcols + w
        se = s * self.pointcols + e
        nw = n * self.pointcols + w
        ne = n * self.pointcols + e

        self.visit_corners(sw, se, nw, ne)
        # while not (self.contained[sw] or self.contained[se] or  self.contained[nw] or self.contained[ne]):
        #     s += 1
        #     n -= 1
        #     w += 1
        #     e -= 1
        #     sw = s * self.pointcols + w
        #     se = s * self.pointcols + e
        #     nw = n * self.pointcols + w
        #     ne = n * self.pointcols + e
        #     self.visit_corners(sw, se, nw, ne)

        # if self.contained[sw] and self.contained[se] and self.contained[nw] and self.contained[ne]:
        self.recursion(w, s, e, n)

    cdef inline void visit_corners(self, Py_ssize_t sw, Py_ssize_t se,  Py_ssize_t nw, Py_ssize_t ne):
        if not self.visited[sw]:
            self.contained[sw] = self.points[sw].intersects(self.footprint)
            self.visited[sw] = True
        if not self.visited[se]:
            self.contained[se] = self.points[se].intersects(self.footprint)
            self.visited[se] = True
        if not self.visited[nw]:
            self.contained[nw] = self.points[nw].intersects(self.footprint)
            self.visited[nw] = True
        if not self.visited[ne]:
            self.contained[ne] = self.points[ne].intersects(self.footprint)
            self.visited[ne] = True

    cdef inline void fill(self, Py_ssize_t w, Py_ssize_t s,  Py_ssize_t e, Py_ssize_t n):
        for r in range(s, n + 1):
            for c in range(w, e + 1):
                self.contained[r * self.pointcols + c] = 1
        self.size += (e - w) * (n - s)

    cdef inline void partial(self, Py_ssize_t w, Py_ssize_t s,  Py_ssize_t e, Py_ssize_t n):
        ...


    cdef recursion(self, Py_ssize_t w, Py_ssize_t s, Py_ssize_t e, Py_ssize_t n):
        sw = s * self.pointcols + w
        se = s * self.pointcols + e
        nw = n * self.pointcols + w
        ne = n * self.pointcols + e

        self.visit_corners(sw, se, nw, ne)

        # TODO: Problem may be, that all of the bounds are not contained
        if  self.contained[sw] and self.contained[se] and self.contained[nw] and self.contained[ne] :
            self.fill(w, s, e, n) # Case 1: Entire box is contained
        elif self.contained[sw] or self.contained[se] or self.contained[nw] or self.contained[ne]:
            # Case 3: Box is partially contained
            if w + 1 == e and s + 1 ==n :
                return # Cannot split further, and the box is not contained, so reject it
            midx = w + e
            midx = midx // 2 + (midx % 2)
            midy = s + n
            midy = midy // 2 + (midy % 2)
            self.recursion(w, s, midx, midy)
            self.recursion(midx, s, e, midy)
            self.recursion(w, midy, midx, n)
            self.recursion(midx, midy, e, n)



cdef class PolygonClaim(ShapeClaim):
    footprint: spatialpandas.geometry.Polygon
    def __init__(
            self,
            footprint: spatialpandas.geometry.Polygon,
            unsigned char length,
            unsigned long lw,
            unsigned long ls,
            unsigned long le,
            unsigned long ln,
    ):
        self.footprint = footprint
        super().__init__(footprint, length, lw, ls, le, ln)

    @classmethod
    def from_shapely(cls, shapely_polygon: shapely.geometry.base.BaseGeometry, unsigned char length):
        return cls(spatialpandas.geometry.Polygon(shapely_polygon), length)

cdef class MultiPolygonClaim(ShapeClaim):
    footprint: spatialpandas.geometry.MultiPolygon
    def __init__(
            self,
            footprint: spatialpandas.geometry.MultiPolygon,
            unsigned char length,
            unsigned long lw,
            unsigned long ls,
            unsigned long le,
            unsigned long ln,
    ):
        self.footprint = footprint
        super().__init__(footprint, length, lw, ls, le, ln)

    @classmethod
    def from_shapely(cls, footprint: shapely.geometry.base.BaseGeometry, unsigned char length):
        return cls(spatialpandas.geometry.MultiPolygon.from_shapely(footprint), length)

cpdef np.ndarray[UINT64, ndim=2] get_claim(
        footprint: shapely.geometry.base.BaseGeometry,
        unsigned char length,
):
    cdef Py_ssize_t size, r, c
    if isinstance(footprint, shapely.geometry.Polygon):
        claim = PolygonClaim(footprint, length)
    elif isinstance(footprint, shapely.geometry.MultiPolygon):
        claim = MultiPolygonClaim(footprint, length)
    else:
        raise ValueError('Unsupported geometry type: %s' % type(footprint))
    size = claim.size

    # cdef np.ndarray[UINT64, ndim=2] longs = np.zeros((size, 2), dtype=np.uint64)
    cdef np.ndarray[UINT64, ndim=2] longs = np.ndarray((size, 2), dtype=np.uint64)
    # print(f'size={size}')
    cdef unsigned long[:, :] longv = longs
    for n in range(claim.size):
        longv[n, 0] = claim.longs[n, 0]
        longv[n, 1] = claim.longs[n, 1]
    return longs

# def generate_claims(footprints: Union[gpd.GeoDataFrame, gpd.GeoSeries]) -> gpd.GeoSeries:
def generate_claims(
        footprints: Union[gpd.GeoDataFrame, gpd.GeoSeries],
        np.ndarray[UINT8, ndim=1] lengths,
):
    cdef Py_ssize_t n, size, i
    cdef unsigned long[:, :] longv
    cdef ShapeClaim claim

    footprints = footprints.to_crs(epsg=4326).geometry
    loc = np.fromiter((
        isinstance(footprint, shapely.geometry.MultiPolygon)
        for footprint in footprints
    ), dtype=bool, count=len(footprints))
    multipolygons = spatialpandas.geometry.MultiPolygonArray.from_geopandas(footprints[loc])
    polygons = spatialpandas.geometry.PolygonArray.from_geopandas(footprints[~loc])

    fw, fs, fe, fn = footprints.geometry.bounds.T.values
    lw = np.ndarray.astype((fw + MAX_LON) * FINAL_LON_PRECISION, dtype=np.uint64)
    ls = np.ndarray.astype((fs + MAX_LAT) * FINAL_LAT_PRECISION, dtype=np.uint64)
    le = np.ndarray.astype((fe + MAX_LON) * FINAL_LON_PRECISION, dtype=np.uint64)
    ln = np.ndarray.astype((fn + MAX_LAT) * FINAL_LAT_PRECISION, dtype=np.uint64)
    # lengths = cfuncs.get_lengths(fw, fs, fe, fn)

    claims = list(map( MultiPolygonClaim, multipolygons, lengths[loc], lw[loc], ls[loc], le[loc], ln[loc] ))
    claims.extend(map( PolygonClaim, polygons, lengths[~loc], lw[~loc], ls[~loc], le[~loc], ln[~loc] ))
    print(f'claims={len(claims)}')
    print(f'count')

    lengths = np.concatenate((lengths[loc], lengths[~loc]))

    sizes = [claim.size for claim in claims]
    count = sum(sizes)
    cdef np.ndarray[UINT64, ndim=1] extension = np.fromiter((
        i
        for i, size in enumerate(sizes)
        for n in range(size)
    ), dtype=np.uint64, count=count)
    lengths = lengths[extension]

    cdef np.ndarray[UINT64, ndim=2] longs = np.ndarray((count, 2), dtype=np.uint64)
    cdef unsigned char[:] lengthv = lengths
    longv = longs

    iloc = np.arange(len(footprints), dtype=np.uint64)
    iloc = iloc[extension]
    cdef unsigned long [:] ilocv = iloc

    n = 0

    for claim, size in zip(claims, sizes):
        for i in range(size):
            longv[n, 0] = claim.longs[i, 0]
            longv[n, 1] = claim.longs[i, 1]
            n += 1

    print(f'n={n}, count={count}')

    # longv = longv[:n, :]
    # lengthv = lengthv[:n]
    # iloc = iloc[:n]
    strings = cfuncs.get_strings(longv[:, 0], longv[:, 1], lengthv)
    index = pd.MultiIndex.from_arrays(( iloc, strings, ), names=('footprint', 'claim'))

    bounds = cfuncs.get_bounds(longv[:, 0], longv[:, 1], lengthv)
    geometry = pygeos.creation.box( bounds[:,0], bounds[:,1], bounds[:,2], bounds[:,3])
    claims = gpd.GeoSeries(
        geometry,
        index=index,
        crs=footprints.crs,
    )
    return claims
