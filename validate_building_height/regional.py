import functools
import re
import warnings
from typing import Generator
from typing import Iterable
from typing import Union, Iterator

import boto3
import geopandas as gpd
import pandas as pd
from botocore import UNSIGNED
from botocore.client import Config
from botocore.handlers import disable_signing
from shapely.geometry import Polygon
from validate_osm.source.resource import StaticRegional, File


# TODO: How do we query for bbox with large, regionally defined datasets?

class MSBuildingFootprints(StaticRegional):
    crs = 'epsg:4326'
    columns = 'geometry'
    name = 'msbf'
    link = ''

    class RegionSouthAmerica(StaticRegional.Region):
        menu = {'Argentina', 'Bolivia', 'Brazil', 'Chile', 'Colombia', 'Ecuador', 'Guyana', 'Paraguay',
                'Peru', 'Uruguay', 'Venezuela', }

        @classmethod
        def __getitem__(self, country: str, *args) -> Generator[File, None, None]:
            if not isinstance(country, str):
                raise TypeError(country)
            url = f'https://minedbuildings.blob.core.windows.net/southamerica/{country}.geojsonl.zip'
            yield File(url=url, path=MSBuildingFootprints.directory / url.rpartition('/')[2])

    class RegionUS(StaticRegional.Region):

        menu = {'United States of America', 'usa'}

        state_names = {"Alaska", "Alabama", "Arkansas", "Arizona", "California", "Colorado",
                       "Connecticut", "DistrictofColumbia", "Delaware", "Florida", "Georgia", "Hawaii",
                       "Iowa", "Idaho", "Illinois", "Indiana", "Kansas", "Kentucky", "Louisiana", "Massachusetts",
                       "Maryland", "Maine", "Michigan", "Minnesota", "Missouri", "Mississippi", "Montana",
                       "North Carolina",
                       "North Dakota", "Nebraska", "New Hampshire", "New Jersey", "New Mexico", "Nevada", "New York",
                       "Ohio", "Oklahoma", "Oregon", "Pennsylvania", "Rhode Island", "South Carolina",
                       "South Dakota", "Tennessee", "Texas", "Utah", "Virginia", "Vermont", "Washington",
                       "Wisconsin", "West Virginia", "Wyoming"}

        @functools.cached_property
        def _states(self) -> gpd.GeoDataFrame:
            gdf = gpd.read_file('https://www2.census.gov/geo/tiger/GENZ2018/shp/cb_2018_us_state_20m.zip')
            gdf = gdf.loc[gdf['NAME'].isin(self.state_names)]
            return gdf

        def __getitem__(self, item: Iterable) -> Generator[File, None, None]:
            # unpack = iter(item)
            # country = next(unpack)
            # states: Union[str, Iterable[str]] = next(unpack, None)
            #
            # if isinstance(country, Polygon):
            #     states = self._states.loc[self._states.geometry.intersects(country), 'NAME']
            # match states:
            #     case str():
            #         states = (states,)
            #     case slice():
            #         states = self.state_names
            #     case None:
            #         raise TypeError(
            #             f"{self.__class__.__name__} received {item};\nAn iterable must be passed"
            #             f" that specifies the states. To receive all states, pass {slice}"
            #         )
            if not isinstance(item, Polygon):
                raise TypeError(item)

            gdf = self._states
            states = gdf.loc[gdf.geometry.intersects(item), 'NAME']

            states: Iterator[str] = (
                str(state).title().replace(' ', '')
                for state in states
            )
            states = (
                'DistrictofColumbia' if state.startswith('Dist') else state
                for state in states
            )
            urls: Iterator[str] = (
                f'https://usbuildingdata.blob.core.windows.net/usbuildings-v2/{state}.geojson.zip'
                for state in states
            )
            yield from (
                File(url=url, path=MSBuildingFootprints.directory / url.rpartition('/')[2])
                for url in urls
            )

    class RegionAustralia(StaticRegional.Region):
        menu = {'Australia'}

        def __getitem__(self, item: str, *args) -> File:
            url = 'https://usbuildingdata.blob.core.windows.net/australia-buildings/Australia_2020-06-21.geojson.zip'
            yield File(url=url, path=MSBuildingFootprints.directory / url.rpartition('/')[2])

    class RegionUgandaTanzania(StaticRegional.Region):
        menu = {'Uganda', 'Tanzania'}

        def __getitem__(self, country: str, *args) -> File:
            if not isinstance(country, str):
                raise TypeError(country)
            url = f'https://usbuildingdata.blob.core.windows.net/tanzania-uganda-buildings/{country}_2019-09-16.zip'
            yield File(url=url, path=MSBuildingFootprints.directory / url.rpartition('/')[2])

    regions = (RegionUS(), RegionSouthAmerica(), RegionAustralia(), RegionUgandaTanzania())

    @functools.cached_property
    def _countries(self) -> gpd.GeoDataFrame:
        countries = gpd.read_file(gpd.datasets.get_path('naturalearth_lowres'))
        countries = countries[countries['name'].isin(self.menu)]
        return countries

    def _files_from_polygon(self, item: Polygon) -> Generator[File, None, None]:
        # TODO: Problem is that we cannot simply query RegionUS; we must specify the states
        #   Thus the solution may be to send the country to the region, and if TypeError is raised, then
        #   we send a Polygon, saying, "Don't want it? Fine! You decide!"
        # TODO: Somehow, self._countries is losing its geometry.
        countries: pd.Series = self._countries.loc[self._countries.geometry.intersects(item), 'name']
        for country in countries:
            try:
                yield from self._menu[country][country]
            except TypeError:
                yield from self._menu[country][item]


class OpenCityData(StaticRegional):
    crs = 'epsg:4979'
    name = 'ocd'
    link = ''

    class RegionUS(StaticRegional.Region):
        menu = {"Alaska", "Alabama", "Arkansas", "Arizona", "California", "Colorado",
                "Connecticut", "DistrictofColumbia", "Delaware", "Florida", "Georgia", "Hawaii",
                "Iowa", "Idaho", "Illinois", "Indiana", "Kansas", "Kentucky", "Louisiana", "Massachusetts",
                "Maryland", "Maine", "Michigan", "Minnesota", "Missouri", "Mississippi", "Montana", "NorthCarolina",
                "North Dakota", "Nebraska", "New Hampshire", "NewJersey", "New Mexico", "Nevada", "NewYork",
                "Ohio", "Oklahoma", "Oregon", "Pennsylvania", "Rhode Island", "South Carolina",
                "SouthDakota", "Tennessee", "Texas", "Utah", "Virginia", "Vermont", "Washington",
                "Wisconsin", "WestVirginia", "Wyoming"}

        def __getitem__(self, item):
            if isinstance(item, str):
                state = item
                counties = None
            elif isinstance(item, Iterable):
                state, counties = item
                if isinstance(counties, str):
                    counties = (counties,)
                elif isinstance(counties, Iterable):
                    counties = list(counties)
                else:
                    raise TypeError(counties)
            else:
                raise TypeError(item)

            s3 = boto3.client('s3', config=Config(signature_version=UNSIGNED))
            resource = boto3.resource('s3')
            resource.meta.client.meta.events.register('choose-signer.s3.*', disable_signing)
            bucket = resource.Bucket('opencitymodel')

            if counties is None:
                yield from (
                    File(
                        url=f'https://opencitymodel.s3.amazonaws.com/{obj.key}',
                        directory=OpenCityData.directory / state /
                                  re.search(r'^.*\/county=([^\/]*).*$', obj.key)[1]
                    )
                    for obj in bucket.objects.filter(Prefix=f'2019-jun/parquet/state={state}/')
                )
            elif isinstance(counties, str):
                county = counties
                yield from (
                    File(
                        url=f'https://opencitymodel.s3.amazonaws.com/{obj.key}',
                        directory=OpenCityData.directory / state / county
                    )
                    for obj in bucket.objects.filter(Prefix=f'2019-jun/parquet/state={state}/county={county}'))
            elif isinstance(counties, Iterable):
                counties = (str(county) for county in counties)
                yield from (
                    File(
                        url=f'https://opencitymodel.s3.amazonaws.com/{obj.key}',
                        directory=OpenCityData.directory / state / county
                    )
                    for county in counties
                    for obj in bucket.objects.filter(Prefix=f'2019-jun/parquet/state={state}/county={county}')
                )
            else:
                raise TypeError(counties)

    regions = (RegionUS(),)

    def __get__(self, instance, owner):
        # TODO: Keep track of which file are being downloaded. Once downloaded, I must postprocess them
        #   and convert the ['fp'] column from string to Polygon. Then I save these file.
        warnings.warn('OCM buildings are compiled from MSBuildingFootprints and OpenStreetMaps')
        raise NotImplementedError

    @functools.cached_property
    def _states(self) -> gpd.GeoDataFrame:
        gdf = gpd.read_file('https://www2.census.gov/geo/tiger/GENZ2018/shp/cb_2018_us_state_20m.zip')
        gdf = gdf.to_crs(4326)
        names: Iterable[str] = gdf['NAME']
        gdf['NAME'] = (name.title().replace(' ', '') for name in names)
        gdf = gdf.loc[gdf['NAME'].isin(self.menu)]
        return gdf

    @functools.cached_property
    def _counties(self) -> gpd.GeoDataFrame:
        gdf = gpd.read_file('https://www2.census.gov/geo/tiger/GENZ2018/shp/cb_2018_us_county_20m.zip')
        gdf: gpd.GeoDataFrame = gdf.set_index('STATEFP')
        gdf = gdf.to_crs(4326)
        return gdf

    def _files_from_polygon(self, item: Polygon) -> Generator[File, None, None]:
        states: gpd.GeoDataFrame = self._states.loc[self._states.geometry.intersects(item)]
        for state, df in states.groupby('NAME').groups.items():
            statefp: int = df['STATEFP'].iloc[0]
            counties = self._counties[statefp]
            state: str
            try:
                yield from self._menu[state][state, counties]
            except TypeError:
                yield from self._menu[state][item]


if __name__ == '__main__':
    msbf = MSBuildingFootprints()
    # x = msbf['tanzania', 'australia']

    x = msbf['Tanzania']
    # x = msbf[['usa', 'illinois'], 'guyana']
