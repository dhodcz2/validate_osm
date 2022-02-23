import abc
from typing import Iterable

import numpy as np
import pandas as pd
import shapely.geometry
from geopandas import GeoDataFrame

from ValidateOSM.source import (
    BBox,
    Source,
    SourceOSM,
    data,
    aggregate,
    enumerative,
    StaticNaive
)


#
# class SelectiveStatic(Static):
#     """
#     https://s3.dualstack.us-east-1.amazonaws.com/opencitymodel/<data-version>/<file-format>/
#     <StateName>/<CountyCodeFIPS>/<StateName>-<CountyCodeFIPS>-<FileNumber>.
#     will probably use this to programatically download the files
#         but how can we be certain that they are of relevance?
#         We iterate across all US counties; those that touch the BBox are considered to be part
#         https://data.world/us-hhs-gov/1456181d-b0eb-4c1c-aa4e-4900fe1cbfa2
#     """
#     sample = """https://s3.dualstack.us-east-1.amazonaws.com/opencitymodel/2019-jun/gml/Arkansas/05053/
#     Arkansas-05053-000.zip"""
#
#     # Direct links seem to be acceptabe as there are no versions other than june-2019; there is no 'home page'
#     #   other than github page itself.
#
#     def __init__(self):
#         # TODO: this is off data.world: US Countries however data.world requires authentication so I am temporarily
#         #   just downloading it and keeping it that way.
#         # TODO: This dataframe may be ill-suited as a it is given as CSV which GeoPandas does not natively support.
#         #   I will either have to replace it or do some extra work
#
#         # TODO: For now pretend that it's a GeoDataFrame that is returned.
#         self.reference: GeoDataFrame = pd.read_csv(
#             "/home/arstneio/PycharmProjects/ValidateOSM/validating_building_height/"
#             "us-hhs-gov-1456181d-b0eb-4c1c-aa4e-4900fe1cbfa2/data/csv_1.csv"
#         )
#
#     def __get__(self, instance, owner):
#         self._instance = instance
#         self._owner = owner
#         bbox = shapely.geometry.Polygon(self._instance.bbox.data.cartesian)
#         reference: GeoDataFrame = self.reference.loc[self.reference.intersects(bbox, align=False)]
#         states: pd.Series
#         ids: pd.Series
#         # Determine the range of ids to query. Either access the page for that state for information,
#         #   or 404 raises StopIteration.
#         urls: Iterable[str] = [
#             f'https://s3.dualstack.us-east-1.amazonaws.com/opencitymodel/2019-jun/gml/{state}/{id}/{state}-{id}-'
#             for state, id in zip(states, ids)
#         ]
#         return urls
#         # TODO: Finish this up
#

class OpenCityModel(Source, abc.ABC):
    name = 'ocm'
    link = 'https://github.com/opencitymodel/opencitymodel'

    @property
    def resource(self) -> GeoDataFrame:
        self.bbox.resource.crs = 'epsg:4979'
        return self.resource

    def height_m(self) -> Iterable[float]:
        return np.nan

    def floors(self):
        return np.nan

    def geometry(self):
        return self.resource.geometry

    def address(self):
        return None

    def bin(self):
        return np.nan

    def timestamp(self):
        return pd.NaT

    def start_date(self):
        return pd.NaT
