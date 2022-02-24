from typing import Iterable

import pandas as pd
from geopandas import GeoDataFrame, GeoSeries
from pandas import DataFrame


def concat(gdfs: Iterable[GeoDataFrame]) -> GeoDataFrame:
    """Workaround because GeoDataFrame.concat returns DataFrame; we want to preserve CRS."""
    crs = {}

    def generator():
        nonlocal gdfs
        gdfs = iter(gdfs)
        gdf = next(gdfs)
        for col in gdf:
            if not isinstance(gdf[col], GeoSeries):
                continue
            gs: GeoSeries = gdf[col]
            crs[col] = gs.crs
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

