import dataclasses
from functools import lru_cache
import base64
import functools
import logging
from typing import Iterable
from typing import Iterator

import bs4
import geopandas as gpd
import pandas as pd
import requests
from geopandas import GeoDataFrame
from geopandas import GeoSeries

from validate_osm import (
    StructFile,
    StructFiles,
    DescriptorStaticRegions,
    logger,
    logged_subprocess
)


# TODO: As per Prof. Miranda's suggestion, try loading from the OSM static PBF file instead of creating queries.


class MicrosoftBuildingFootprints(DescriptorStaticRegions):
    crs = 'epsg:4326'
    name = 'msbf'
    link = 'https://github.com/Microsoft/USBuildingFootprints'

    @functools.cached_property
    def countries(self) -> GeoDataFrame:
        countries = gpd.read_file(gpd.datasets.get_path('naturalearth_lowres'))
        countries = countries.set_index('name')['geometry'].to_crs(4326)
        return countries

    def __delete__(self, instance):
        if 'countries' in self.__dict__:
            del self.countries
        super(MicrosoftBuildingFootprints, self).__delete__(instance)

    class RegionSouthAmerica(DescriptorStaticRegions.StaticRegion):
        @functools.cached_property
        def geometry(self) -> GeoSeries:
            return self.resource.countries[[
                'Argentina', 'Bolivia', 'Brazil', 'Chile', 'Colombia', 'Ecuador', 'Guyana', 'Paraguay',
                'Peru', 'Uruguay', 'Venezuela',
            ]]

        def urls(self, names: Iterable[str]) -> Iterator[str]:
            yield from (
                f'https://usbuildingdata.blob.core.windows.net/usbuildings-v2/{name}.geojson.zip'
                for name in names
            )

    class RegionUS(DescriptorStaticRegions.StaticRegion):
        @functools.cached_property
        def geometry(self) -> GeoSeries:
            geometry = gpd.read_file('https://www2.census.gov/geo/tiger/GENZ2018/shp/cb_2018_us_state_20m.zip')
            geometry['NAME'] = [
                name.replace(' ', '')
                for name in geometry['NAME']
            ]
            geometry = geometry.set_index('NAME').geometry
            geometry = geometry.drop('PuertoRico')
            return geometry

        def urls(self, names: Iterable[str]) -> Iterator[StructFile]:
            yield from (
                f'https://usbuildingdata.blob.core.windows.net/usbuildings-v2/{name}.geojson.zip'
                for name in names
            )

    class RegionAustralia(DescriptorStaticRegions.StaticRegion):

        @functools.cached_property
        def geometry(self) -> GeoSeries:
            return self.resource.countries[['Australia']]

        def urls(self, names: Iterable[str]) -> Iterator[str]:
            yield 'https://usbuildingdata.blob.core.windows.net/australia-buildings/Australia_2020-06-21.geojson.zip'

    class RegionUgandaTanzania(DescriptorStaticRegions.StaticRegion):
        @functools.cached_property
        def geometry(self) -> GeoSeries:
            return self.resource.countries[['Uganda', 'Tanzania']]

        def urls(self, names: Iterable[str]) -> Iterator[str]:
            yield from (
                f'https://usbuildingdata.blob.core.windows.net/tanzania-uganda-buildings/{name}_2019-09-16.zip'
                for name in names
            )


def create_onedrive_directdownload(onedrive_link):
    # thanks https://towardsdatascience.com/onedrive-as-data-storage-for-python-project-2ff8d2d3a0aa
    data_bytes64 = base64.b64encode(bytes(onedrive_link, 'utf-8'))
    data_bytes64_String = data_bytes64.decode('utf-8').replace('/', '_').replace('+', '-').rstrip("=")
    resultUrl = f"https://api.onedrive.com/v1.0/shares/u!{data_bytes64_String}/root/content"
    return resultUrl


class MicrosoftBuildingFootprints2017(DescriptorStaticRegions):
    crs = 'epsg:4326'
    name = 'msbf2017'
    link = 'https://wiki.openstreetmap.org/wiki/Microsoft_Building_Footprint_Data'

    class California(DescriptorStaticRegions.StaticRegion):
        @functools.cached_property
        def url(self) -> dict[str, str]:
            res = requests.get('https://wiki.openstreetmap.org/wiki/Microsoft_Building_Footprint_Data')
            res.raise_for_status()
            soup = bs4.BeautifulSoup(res.text, 'html.parser')
            x = soup.find_all('table', {'class': 'wikitable'})[1]
            rows = [
                row.find_all('td')
                for row in x.find_all('tr')
            ]
            return {
                row[0].text.replace('\n', ''): create_onedrive_directdownload(row[3].find_next('a')['href'])
                for row in rows
                if len(row)
            }

        def urls(self, names: Iterable[str]) -> Iterator[str]:
            url = self.url
            yield from (
                url[name]
                for name in names
                if name in url
            )

        @functools.cached_property
        def geometry(self) -> GeoSeries:
            geometry = gpd.read_file(
                'https://data.ca.gov/dataset/e212e397-1277-4df3-8c22-40721b095f33/resource/'
                '436fc714-831c-4070-b44b-b06dcde6bf18/download/ca-places-boundaries.zip',
            )
            geometry: GeoSeries = geometry.set_index('NAME').gdf
            bay = ['San Francisco', 'Cupertino', 'San Jose', 'Berkeley', 'Fremont']
            crs = geometry.crs
            geometry['Bay Area (needs to be further broken apart)'] = geometry[bay].unary_union
            geometry['Gilroy, Morgan Hill, Hollister'] = geometry[['Gilroy', 'Morgan Hill', 'Hollister']].unary_union
            geometry.crs = crs
            geometry = geometry.to_crs('epsg:4326')
            return geometry

    class NotCalifornia(DescriptorStaticRegions.StaticRegion):
        @functools.cached_property
        def url(self) -> dict[str, str]:
            res = requests.get('https://wiki.openstreetmap.org/wiki/Microsoft_Building_Footprint_Data')
            res.raise_for_status()
            soup = bs4.BeautifulSoup(res.text, 'html.parser')
            x = soup.find_all('table', {'class': 'wikitable'})[0]
            rows = [
                row.find_all('td')
                for row in x.find_all('tr')
            ]
            return {
                row[0].text.replace('\n', ''): create_onedrive_directdownload(row[2].find_next('a')['href'])
                for row in rows
                if len(row)
            }

        @functools.cached_property
        def geometry(self) -> GeoSeries:
            geometry = gpd.read_file('https://www2.census.gov/geo/tiger/GENZ2018/shp/cb_2018_us_state_20m.zip')
            geometry['name'] = [
                name.replace(' ', '')
                for name in geometry['NAME']
            ]
            geometry = geometry.set_index('name')['geometry']
            geometry = geometry.to_crs(4326)

            geometry = geometry[geometry.index.drop('California')]
            return geometry

        def urls(self, names: Iterable[str]) -> Iterator[str]:
            url = self.url
            yield from (
                url[name]
                for name in names
                if name in url
            )

    def __getitem__(self, item):
        for file in super(MicrosoftBuildingFootprints2017, self).__getitem__(item):
            # TODO: This can be done more cleanly
            name = requests.head(file.url).headers['Location'].rpartition('/')[2]
            values = {
                key: file.__dict__[key]
                for key in file.__dataclass_fields__.keys()
            }
            values['name'] = name
            new = StructFile(**values)
            new.resource = file.resource.parent / name
            yield new


class OpenCityModel(DescriptorStaticRegions):
    crs = 'epsg:4979'
    name = 'ocm'

    @dataclasses.dataclass(repr=False)
    class StructFiles(StructFiles):
        def __iter__(self) -> Iterator[pd.DataFrame]:
            for file in self.files:
                logger.debug(f'reading {file.name}')
                yield pd.read_parquet(file.resource, columns=['fp', 'ubid', 'height', 'fp_source'])

        def load_resource(self) -> GeoDataFrame:
            df = pd.concat(self)
            return gpd.GeoDataFrame({
                'geometry': GeoSeries.from_wkt(df['fp'], crs='epsg:4979'),
                'ubid': df['ubid'],
                'height': df['height'],
                'fp_source': df['fp_source']
            })

    # TODO: Returns StructFile instead of StructFiles?
    class RegionUS(DescriptorStaticRegions.StaticRegion):
        @functools.cached_property
        def geometry(self) -> GeoSeries:
            states = gpd.read_file('https://www2.census.gov/geo/tiger/GENZ2018/shp/cb_2018_us_state_20m.zip')
            state_names = {
                fp: name.replace(' ', '')
                for fp, name in states[['STATEFP', 'NAME']].values
            }
            counties = gpd.read_file('https://www2.census.gov/geo/tiger/GENZ2018/shp/cb_2018_us_county_20m.zip')
            gdf = GeoDataFrame({
                'geometry': counties['geometry'].to_crs('epsg:4326'),
                'county': (
                    f'{statefp}{countyfp}'
                    for statefp, countyfp in counties[['STATEFP', 'COUNTYFP']].values
                ),
                'state': (
                    state_names[statefp]
                    for statefp in counties['STATEFP']
                )
            })
            gdf = gdf.set_index(['state', 'county'])
            return gdf

        def urls(self, names: Iterable[str]) -> Iterator[str]:
            from botocore.client import Config
            from botocore import UNSIGNED
            from botocore.handlers import disable_signing
            import boto3
            s3 = boto3.client('s3', config=Config(signature_version=UNSIGNED))
            resource = boto3.resource('s3')
            resource.meta.client.meta.events.register('choose-signer.s3.*', disable_signing)
            bucket = resource.Bucket('opencitymodel')

            for state, county in names:
                # for obj in bucket.objects.filter(Prefix=f'2019-jun/parquet/state={state}/county={county}'):
                #     yield f'https://opencitymodel.s3.amazonaws.com/{obj.key}'
                yield (
                    f'https://opencitymodel.s3.amazonaws.com/{obj.key}'
                    for obj in bucket.objects.filter(Prefix=f'2019-jun/parquet/state={state}/county={county}')
                )

        def __getitem__(self, item):
            for file in super(OpenCityModel.RegionUS, self).__getitem__(item):
                # To retain all the attributes, to avoid annoyance for any modifications to StructFiles
                yield OpenCityModel.StructFiles(**{
                    key: file.__dict__[key]
                    for key in file.__dataclass_fields__.keys()
                })
