import abc
import geopandas as gpd

import datetime
from typing import Iterable, Union

# from ValidateOSM.data import Source, Identifier, Validating, NonEssentialData, EssentialData, Haystack
import numpy as np
import pandas as pd
import shapely.geometry.base
from geopandas import GeoDataFrame

from ValidateOSM.config import ROOT
# from validating_building_height.base import Height, NeedlesHeight
from validating_building_height.base import (
    Height, NeedlesHeight
)
from ValidateOSM.source.source import (
    _BBox, BBox,
)
from ValidateOSM.source import (
    Source,
    SourceOSM,
    data,
    aggregate,
    enumerative,
)


class NewYork(Source, abc.ABC):
    @data.identifier('Int64')
    @abc.abstractmethod
    def bin(self) -> Iterable[int]:
        """The New York Building Identification Number"""

    bbox = BBox([40.70010232505462, -74.01998471326246, 40.750102666482995, -73.96432847885934])


class NewYorkBuildingHeightNeedles(NeedlesHeight, NewYork):
    @enumerative(int)
    def bin(self):
        return (
            element.tag('nycdoitt:bin')
            for element in self.source
        )


class NewYork3DModel(NewYork, Height):
    def __init__(self):
        self.bbox.raw.crs = 2263
        self.bbox.raw.flip = True

    @property
    def raw(self) -> GeoDataFrame:
        path = ROOT / 'validating_building_height' / 'static' / 'DA_WISE_Multipatch'
        return self.from_file(path)

    @classmethod
    @property
    def source_information(cls) -> str:
        return 'https://www1.nyc.gov/site/doitt/initiatives/3d-building.page'

    @classmethod
    @property
    def name(cls) -> str:
        return '3dm'

    def height_m(self) -> Iterable[float]:
        METERS_PER_FOOT = .3048
        for geometry in self.raw.geometry:
            heights = {
                geom.exterior.coords[0][2]
                for geom in geometry.geoms
            }
            yield (max(heights) - min(heights)) * METERS_PER_FOOT

    def floors(self) -> Iterable[float]:
        return np.nan

    def geometry(self) -> Iterable[shapely.geometry.base.BaseGeometry]:
        return self.raw.geometry

    def address(self) -> Iterable[str]:
        return None

    def start_date(self) -> Iterable[datetime.datetime]:
        return None

    def timestamp(self):
        return datetime.datetime(2014, 1, 1)

    def bin(self):
        return self.raw['BIN']


class NewYorkOpenCityModel(NewYork, Height):
    def __init__(self):
        self.bbox.raw.crs = 'epsg:4979'

    @classmethod
    @property
    def raw(self) -> GeoDataFrame:
        path = ROOT / 'validating_building_height' / 'static' / 'NewYork-36061-000.gml'
        return self.read_file(path)

    @classmethod
    @property
    def name(cls) -> str:
        return 'ocm'

    @classmethod
    @property
    def source_information(cls):
        return 'https://github.com/opencitymodel/opencitymodel'

    def height_m(self) -> Iterable[float]:
        return np.nan

    def floors(self):
        return np.nan

    def geometry(self):
        return self.raw.geometry

    def address(self):
        return None

    def bin(self):
        return np.nan

    def timestamp(self):
        return pd.NaT

    def start_date(self):
        return pd.NaT


class NewYorkLOD(NewYork, Height):
    def __init__(self):
        self.bbox.raw.crs = None

    @property
    @classmethod
    def raw(self) -> GeoDataFrame:
        path = ROOT / 'validating_building_height' / 'static' / 'NYC_Buildings_LoD2_CityGML.gml'
        return self.read_file(path)

    @classmethod
    @property
    def name(cls) -> str:
        return 'lod'

    @classmethod
    @property
    def source_information(cls):
        return 'https://www.asg.ed.tum.de/gis/projekte/new-york-city-3d/#c753'

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
