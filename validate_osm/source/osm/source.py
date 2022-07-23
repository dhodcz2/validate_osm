import geopandas as gpd
from geopandas import GeoDataFrame, GeoSeries
from pandas import Series, DataFrame

import pandas as pd

from ..data import DecoratorData, column
from .handler import BaseHandler
from .resource_ import ResourceOsmium
from ..source import Source


# TODO: How to group data?

class SourceOSM(Source):
    resource: ResourceOsmium | BaseHandler = ResourceOsmium()

    @column
    def geometry(self):
        return GeoSeries(self.resource.geometry, crs=4326)

    @column
    def timestamp(self):
        return self.resource.timestamp

    @column
    def group(self):
        return self.resource.groups

    # @DecoratorData
    # def data(self):
    #     data = super(SourceOSM, self).data()
    #     # TODO: Is it possible to group the data returned by DescriptorData?
    #

