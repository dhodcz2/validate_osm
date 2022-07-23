import geopandas as gpd
from geopandas import GeoDataFrame, GeoSeries
from pandas import Series, DataFrame


class Comparison(GeoDataFrame):
    _constructor = GeoDataFrame
    _metadata = GeoDataFrame._metadata + ['sources']


