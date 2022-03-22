import abc
import geopandas as gpd

import datetime
from typing import Iterable, Union

# from validate_osm.data import Source, Identifier, Validating, NonEssentialData, EssentialData, Haystack
import numpy as np
import pandas as pd
import shapely.geometry.base
from geopandas import GeoDataFrame

from validate_osm.config import ROOT
# from validate_building_height.base import Height, NeedlesHeight
# from validating_building_height.base import (
#     Height, NeedlesHeight
# )
from validate_building_height.base import Height, HeightOSM
from validate_osm.source.source import (
    BBox, BBox,
)
from validate_osm.source import (
    Source,
    SourceOSM,
    data,
    aggregate,
    enumerative,
)
from validate_osm.source.resource import StaticBase


class NewYork(Source, abc.ABC):
    bbox = BBox([40.70010232505462, -74.01998471326246, 40.750102666482995, -73.96432847885934])
    @data.identifier('Int64')
    @abc.abstractmethod
    def bin(self) -> Iterable[int]:
        """The New York Building Identification Number"""


class NewYorkBuildingHeightNeedles(HeightOSM, NewYork):
    @enumerative(int)
    def bin(self):
        return (
            element.tag('nycdoitt:bin')
            for element in self.source
        )


class NewYork3DModel(NewYork, Height):
    resource = StaticBase(
        uri="https://www1.nyc.gov/site/doitt/initiatives/3d-building.page",
        crs=2263,
        flipped=True,
        unzipped='DA_WISE_Multipatch'
    )
    name = '3dm'
    link = 'https://www1.nyc.gov/site/doitt/initiatives/3d-building.page'

    @property
    def resource(self) -> GeoDataFrame:
        # self.bbox.raw.crs = 2263
        # self.bbox.raw.flip = True
        return self.resource

    def height_m(self) -> Iterable[float]:
        METERS_PER_FOOT = .3048
        for geometry in self.resource.geometry:
            heights = {
                geom.exterior.coords[0][2]
                for geom in geometry.geoms
            }
            yield (max(heights) - min(heights)) * METERS_PER_FOOT

    def floors(self) -> Iterable[float]:
        return np.nan

    def geometry(self) -> Iterable[shapely.geometry.base.BaseGeometry]:
        return self.resource.geometry

    def address(self) -> Iterable[str]:
        return None

    def start_date(self) -> Iterable[datetime.datetime]:
        return None

    def timestamp(self):
        return datetime.datetime(2014, 1, 1)

    def bin(self):
        return self.resource['BIN']


class NewYorkOpenCityModel(NewYork, Height):
    # TODO: The OpenCityModel is HUGE. I need to put more time into fully utilizing this dataset.
    #   This uses:
    #       MS USBuildingFootprints
    #       MS Building Footprints (2017)
    #       Open Street Map
    #       LA County LARIAC4 (2014)

    # TODO: Is it problematic that we are validating OpenStreetMaps with a source that contains Open Street Maps itself?
    #   Can we filter out the OSM entries and still use the dataset?


    # TODO: UBID is the standard for uniquely identifying building footprints on the Earth.
    #   UBID for a building footprint has five components:
    #   1   Open Location Code for the geometric center of mass (centroid) of the building footprint.
    #   2   The distance to tne northern extent of the bounding box for the building foot in Open Location Code units
    #   3   The distance to the eastern extent ...
    #   4   The distance to the southern extent ...
    #   5   The distance to the western extent ...

    # TODO: Temporary placeholder to just leverage one source that is particularly
    # TODO: This Static file is zipped with one file. Be certain this works without specification of unzipped contents
    resource = StaticBase(
        url='https://s3.dualstack.us-east-1.amazonaws.com/opencitymodel/2019-jun/gml/NewYork/36061/'
            'NewYork-36061-000.zip',
        crs='epsg:4979'
    )

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


class NewYorkLOD(NewYork, Height):
    link = 'https://www.asg.ed.tum.de/gis/projekte/new-york-city-3d/#c753'
    name = 'lod'
    resource = StaticBase(
        url='http://www.3dcitydb.net/3dcitydb/fileadmin/public/datasets/NYC/NYC_buildings_CityGML_LoD2/'
            'NYC_Buildings_LoD2_CityGML.zip',
        unzipped='NYC_Buildings_LoD2_CityGML.gml',
        crs=None,
    )

    @property
    def resource(self) -> GeoDataFrame:
        self.bbox.resource.crs = None
        return self.resource

    def height_m(self) -> Union[pd.Series, Iterable]:
        ...

    def geometry(self):
        ...

    def address(self):
        ...

    def bin(self):
        ...

    def timestamp(self) -> Iterable[datetime.datetime]:
        ...

    def start_date(self):
        ...

    def floors(self) -> Iterable[float]:
        ...
