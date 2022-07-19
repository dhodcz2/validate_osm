import geopandas as gpd
import datetime
from typing import Iterable, Union
import numpy as np
import pandas as pd
import shapely.geometry.base
from geopandas import GeoDataFrame
from validate_building_height.building import Height
from validate_building_height.source import HeightOSM
from validate_osm import Source, SourceOSM
from validate_osm import DescriptorStaticNaive, StructFile


class NewYork3DModel(Height):
        resource = DescriptorStaticNaive(
            file=StructFile(
                url="http://maps.nyc.gov/download/3dmodel/DA_WISE_Multipatch.zip",
                source=None,
                name='DA_WISE_Multipatch',
            ),
            crs=2263,
            flipped=True,
            unzipped='DA_WISE_Multipatch',
            name='3dm',
            link = 'https://www1.nyc.gov/site/doitt/initiatives/3d-building.page'
        )


class NewYorkLOD(Height):
    resource = DescriptorStaticNaive(
        file=StructFile(
            url='http://www.3dcitydb.net/3dcitydb/fileadmin/public/datasets/NYC/NYC_buildings_CityGML_LoD2/'
                'NYC_Buildings_LoD2_CityGML.zip',
            source=None,
            name=None
        ),
        crs=None,
        name='lod',
        link='https://www.asg.ed.tum.de/gis/projekte/new-york-city-3d/#c753'
    )

    def height_m(self) -> Union[pd.Series, Iterable]:
        ...

    def geometry(self):
        return self.resource.geometry

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
