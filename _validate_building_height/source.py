import datetime
from typing import Iterable

import numpy as np
import pandas as pd
import shapely
from shapely.geometry import box

from validate_building_height.building import Height, BuildingSourceOSM
from validate_building_height.resource_ import (
    MicrosoftBuildingFootprints,
    MicrosoftBuildingFootprints2017,
    OpenCityModel
)
from validate_osm import *


class HeightMicrosoftBuildingFootprints(Height):
    resource = MicrosoftBuildingFootprints()

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


class HeightMicrosoftBuildingFootprints2017(Height):
    resource = MicrosoftBuildingFootprints2017()

    def geometry(self):
        return self.resource.geometry

    def timestamp(self):
        ...

    def address(self):
        ...

    def height_m(self):
        return self.resource['Height']

    def start_date(self):
        ...

    def floors(self):
        ...


class HeightOpenCityModel(Height):
    resource = OpenCityModel()

    def geometry(self):
        return self.resource.geometry

    def timestamp(self):
        ...

    def address(self):
        ...

    def height_m(self):
        return self.resource['height']

    def start_date(self):
        ...

    def floors(self):
        ...


class HeightOSM(BuildingSourceOSM, Height):
    bbox = True

    @DecoratorEnumerative(float)
    def floors(self):
        yield from (
            element.tag('building:levels')
            for element in self.resource
        )

    @DecoratorEnumerative(float)
    def height_m(self):
        # TODO: How do we convert to meters if it says feet?
        for element in self.resource:
            # height = element.tag('height')
            # if height is not None and height[-1] == "'":
            #     yield float(height[:-1]) * .3048
            # else:
            #     yield height
            yield element.tag('height')


class HeightChicagoBuildingFootprints(Height):
    bbox = BBox((
        41.20771257335822, -88.71312626050252, 42.6664014741069, -86.88939577195931
    ), crs='epsg:4326')

    resource = DescriptorStaticNaive(
        file=StructFile(
            url='https://data.cityofchicago.org/api/geospatial/hz9b-7nh8?method=export&format=GeoJSON',
            source=None,
            name='Building Footprints (current).geojson'
        ),
        crs=4326,
        name='cbf',
        link='https://data.cityofchicago.org/Buildings/Building-Footprints-current-/hz9b-7nh8',
    )

    def geometry(self) -> Iterable[shapely.geometry.base.BaseGeometry]:
        return self.resource.geometry.reset_index(drop=True)

    def address(self) -> Iterable[object]:
        def generator():
            for num, dir, name, type in self.resource.loc[:, ['t_add1', 'pre_dir1', 'st_name1', 'st_type1']].values:
                if not (name and type):
                    yield None
                    continue
                res: str = num
                for frag in (dir, name, type):
                    if frag is not None:
                        res += f' {frag}'
                yield res

        yield from generator()

    # TODO: TypeError: <U3 cannot be converted to a FloatingDtype
    def floors(self) -> Iterable[float]:
        # yield from (
        #     np.nan if floors == 0
        #     else floors
        #     for floors in self.resource['stories']
        # )
        def gen():
            for value in self.resource['stories']:
                try:
                    v = float(value)
                except (TypeError, ValueError):
                    yield np.nan
                else:
                    if v == 0:
                        yield np.nan
                    else:
                        yield v

        yield from gen()

    def bldg_id(self) -> Iterable[int]:
        return self.resource['bldg_id'].reset_index(drop=True)

    def height_m(self) -> Iterable[float]:
        return np.nan

    def timestamp(self) -> Iterable[datetime.datetime]:
        # TODO: No longer uses date_bld_2; too tired to piece together year and everything
        # return self.resource['date_bld_2'].reset_index(drop=True)
        return pd.NaT

    def start_date(self) -> Iterable[datetime.datetime]:
        # yield from (
        #     pd.NaT if year == 0
        #     else datetime.datetime(int(year), 1, 1)
        #     for year in self.resource['year_built']
        # )
        def gen():
            for value in self.resource['year_built']:
                try:
                    v = int(value)
                except (TypeError, ValueError):
                    yield pd.NaT
                else:
                    if v == 0:
                        yield pd.NaT
                    else:
                        try:
                            yield datetime.datetime(v, 1, 1)
                        except ValueError:
                            yield pd.NaT

        yield from gen()
