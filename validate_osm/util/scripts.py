import logging
from python_log_indenter import IndentedLoggerAdapter
from typing import Iterable, Iterator, Union
import time
from contextlib import contextmanager

import pandas as pd
from geopandas import GeoDataFrame, GeoSeries
from pandas import DataFrame

# def concat(gdfs: Iterable[GeoDataFrame]) -> GeoDataFrame:
#     """Workaround because GeoDataFrame.concat returns DataFrame; we want to preserve CRS."""
#     crs = {}
#
#     def generator():
#         nonlocal gdfs
#         gdfs = iter(gdfs)
#         gdf = next(gdfs)
#         for col in gdf:
#             if not isinstance(gdf[col], GeoSeries):
#                 continue
#             gs: GeoSeries = gdf[col]
#             crs[col] = gs.crs
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
LoggedSubprocessAdapter: Union[IndentedLoggerAdapter, 'LoggedSubprocessAdapter']


"""
INFO -  
"""
@contextmanager
def logged_subprocess(logger: IndentedLoggerAdapter, message: str, level=logging.INFO, timed=True):
    match level:
        case logging.DEBUG:
            logger.debug(message)
        case logging.INFO:
            logger.info(message)
        case logging.WARN:
            logger.warning(message)
        case logging.ERROR:
            logger.error(message)
        case logging.CRITICAL:
            logger.critical(message)

    logger.add()
    t = time.time()
    yield
    t = (time.time() - t) / 60
    message = f'{t:.1f} minutes'
    if timed:
        match level:
            case logging.DEBUG:
                logger.debug(message)
            case logging.INFO:
                logger.info(message)
            case logging.WARN:
                logger.warning(message)
            case logging.ERROR:
                logger.error(message)
            case logging.CRITICAL:
                logger.critical(message)
    logger.sub()


# class LoggedSubprocessAdapter:
#     __new__: Union[IndentedLoggerAdapter, 'LoggedSubprocessAdapter']
#     def __init__(self, logger: IndentedLoggerAdapter):
#         self._logger = logger
#
#     def __getattribute__(self, item):
#         if hasattr(self, item):
#             return super(LoggedSubprocessAdapter, self).__getattribute__(item)
#         return self._logger.__class__.__getattribute__(self._logger, item)
#
#     @contextmanager
#     def logged_subprocess(logger: IndentedLoggerAdapter, header: str, level=logging.INFO):
#         match level:
#             case logging.DEBUG:
#                 logger.debug(header)
#             case logging.INFO:
#                 logger.info(header)
#             case logging.WARN:
#                 logger.warning(header)
#             case logging.ERROR:
#                 logger.error(header)
#             case logging.CRITICAL:
#                 logger.critical(header)
#
#
#         logger.add()
#         t = time.time()
#         yield
#         t = (time.time() - t) / 60
#         message = logger.info(f'{t:.1f} minutes')
#         match level:
#             case logging.DEBUG:
#                 logger.debug(message)
#             case logging.INFO:
#                 logger.info(message)
#             case logging.WARN:
#                 logger.warning(message)
#             case logging.ERROR:
#                 logger.error(message)
#             case logging.CRITICAL:
#                 logger.critical(message)
#         logger.sub()
# LoggedSubprocessAdapter: Union[IndentedLoggerAdapter, 'LoggedSubprocessAdapter']
#

def concat(gdfs: Iterator[GeoDataFrame]) -> GeoDataFrame:
    crs = {}

    def generator():
        gdf = next(gdfs)
        for col in gdf:
            series = gdf[col]
            if isinstance(series, GeoSeries):
                crs[col] = series.crs
        yield gdf
        yield from gdfs

    result: DataFrame = pd.concat(generator())
    result: GeoDataFrame = GeoDataFrame({
        col: (
            result[col] if col not in crs
            else GeoSeries(result[col], crs=crs[col])
        )
        for col in result
    })
    return result
