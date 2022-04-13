import logging
from typing import Iterator, Iterable

logger = logging.getLogger(__name__.partition('.')[0])

# logger = logging.g

import pandas as pd
from geopandas import GeoDataFrame, GeoSeries
from pandas import DataFrame


# def concat(gdfs: Iterable[GeoDataFrame]) -> GeoDataFrame:
#     gdfs = iter(gdfs)
#     crs = {}
#
#     def generator():
#         gdf = next(gdfs)
#         for col in gdf:
#             series = gdf[col]
#             if isinstance(series, GeoSeries):
#                 crs[col] = series.crs
#         yield gdf
#         yield from gdfs
#
#     result: DataFrame = pd.concat(generator())
#     result: GeoDataFrame = GeoDataFrame({
#         col: (
#             result[col] if col not in crs
#             else GeoSeries(result[col], crs=crs[col])
#         )
#         for col in result
#     })
#     return result
#

def concat(gdfs: Iterable[GeoDataFrame]) -> GeoDataFrame:
    gdfs = iter(gdfs)
    crs = {}

    def gen() -> Iterator[GeoDataFrame]:
        gdf = next(gdfs)
        for col in gdf:
            if isinstance(gdf[col], GeoSeries):
                crs[col] = gdf[col].crs
        yield gdf
        yield from gdfs

    result: DataFrame = pd.concat(gen())
    result: GeoDataFrame = GeoDataFrame({
        col: (
            result[col] if col not in crs
            else
            GeoSeries(result[col], crs=crs[col])
        )
        for col in result
    })
    return result
