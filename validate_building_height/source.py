import datetime
import functools
from typing import Iterable

import numpy as np
import shapely
from geopandas import GeoDataFrame

from validate_building_height.building import Height, BuildingSourceOSM
from validate_osm.source import *
from validate_osm.source.resource import File


class SourceMSBuildingFootprints(Height):
    from validate_building_height.regional import MSBuildingFootprints
    resource = MSBuildingFootprints()

    def geometry(self):
        return self.resource['geometry']

    def timestamp(self):
        return datetime.datetime(2020, 6, 10)

    def address(self):
        ...

    def height_m(self):
        ...

    def start_date(self):
        ...

    def floors(self):
        ...


# class SourceOpenCityData(Height):
#     from validate_building_height.regional import OpenCityData
#     resource = OpenCityData()
#
#     def geometry(self):
#         ...
#
#     def timestamp(self):
#         ...
#
#     def address(self):
#         ...
#
#     def height_m(self):
#         ...
#
#     def start_date(self):
#         ...
#
#     def floors(self):
#         ...
#

class HeightOSM(BuildingSourceOSM, Height):
    @enumerative(float)
    def floors(self):
        yield from (
            element.tag('building:levels')
            for element in self.resource
        )

    @enumerative(float)
    def height_m(self):
        yield from (
            element.tag('height')
            for element in self.resource
        )


class HeightChicagoBuildingFootprints(Height):
    name = 'cbf'
    link = 'https://data.cityofchicago.org/Buildings/Building-Footprints-current-/hz9b-7nh8'
    resource = StaticNaive(
        files=File(
            url='https://data.cityofchicago.org/api/geospatial/hz9b-7nh8?method=export&format=GeoJSON',
            path=StaticNaive.directory / 'Building Footprints (current).geojson',
        ),
        crs=4326
    )

    @functools.cached_property
    def resource(self) -> GeoDataFrame:
        return self.resource

    def geometry(self) -> Iterable[shapely.geometry.base.BaseGeometry]:
        return self.resource.geometry.reset_index(drop=True)

    def address(self) -> Iterable[object]:
        def generator():
            for num, dir, name, type in self.resource.loc[:, ['t_add1', 'pre_dir1', 'st_name1', 'st_type1']]:
                if not (name and type):
                    yield None
                    continue
                res: str = num
                for frag in (dir, name, type):
                    if frag is not None:
                        res += f' {frag}'
                yield res

        yield from generator()

    def floors(self) -> Iterable[float]:
        yield from (
            np.nan if floors == 0
            else floors
            for floors in self.resource['stories']
        )

    def bldg_id(self) -> Iterable[int]:
        return self.resource['bldg_id'].reset_index(drop=True)

    def height_m(self) -> Iterable[float]:
        return np.nan

    def timestamp(self) -> Iterable[datetime.datetime]:
        return self.resource['date_bld_2'].reset_index(drop=True)

    def start_date(self) -> Iterable[datetime.datetime]:
        yield from (
            pd.NaT if year == 0
            else datetime.datetime(int(year), 1, 1)
            for year in self.resource['year_built']
        )
