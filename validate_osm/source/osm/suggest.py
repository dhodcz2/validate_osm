import concurrent.futures
import io
import itertools
import re
from concurrent.futures import Future
from functools import cached_property
from pathlib import Path
from typing import Iterator, Iterable

import numpy as np
import pandas as pd
import pyrosm
import requests
import shapely.geometry
from geopandas import GeoDataFrame
from geopandas import GeoSeries
from pandas import IndexSlice as idx
from pandas.core.indexing import _LocIndexer
from pyrosm.data import Africa
from pyrosm.data import Brazil
from shapely.geometry import MultiPolygon, Polygon
from shapely.geometry.base import BaseGeometry


# TODO: .poly files are not always available; raise an exception telling user to try again later

class CallableConstructPoly:
    def __init__(self):
        self._directory = Path(__file__).parent / 'poly'
        self._directory.mkdir(exist_ok=True, parents=True)

    @staticmethod
    def __extract(file: str, response: requests.Response):
        with open(file, 'w') as f:
            try:
                response.raise_for_status()
            except requests.exceptions.HTTPError as e:
                raise
            else:
                f.write(response.text)

    @staticmethod
    def __transform(textio: io.TextIOWrapper) -> Polygon | MultiPolygon:
        text = textio.read().replace('\n', '')
        matches = re.findall(r'(?<=\d)(.*?)END', text)
        # if not matches:
        #     raise ValueError(f'No polygons found in {text}')
        if not matches:
            raise ValueError(f'No polygons found')
        splits = list(map(str.split, matches))
        lons = (np.array(split[::2], dtype=np.float64) for split in splits)
        lats = (np.array(split[1::2], dtype=np.float64) for split in splits)
        polygons = map(Polygon, map(zip, lons, lats))
        if len(matches) == 1:
            return next(polygons)
        return MultiPolygon(polygons)

    def __call__(self, poly: np.ndarray | Iterable[str]) -> np.ndarray:
        directory = self._directory
        files = [
            directory / poly.rpartition('/')[2]
            for poly in poly
        ]
        compress = [
            not file.exists()
            for file in files
        ]
        session = requests.Session()
        with concurrent.futures.ThreadPoolExecutor() as threads:
            futures: dict[Future, str] = {
                threads.submit(session.get, url): url
                for url in itertools.compress(poly, compress)
            }
            for _ in concurrent.futures.as_completed(futures):
                ...
        exceptions = list(map(Future.exception, futures))
        if any(exceptions):
            raise requests.exceptions.HTTPError(
                f'Unable to download {sum(exceptions)} files; the .poly files available on GeoFabrik come in and out; '
                f'available files are locally cached. Try again later.'
            )

        with concurrent.futures.ThreadPoolExecutor() as threads:
            threads.map(
                self.__extract,
                itertools.compress(files, compress),
                map(Future.result, itertools.compress(futures, compress))
            )
        with concurrent.futures.ThreadPoolExecutor() as threads:
            textio: Iterator = threads.map(lambda file: open(file, 'r'), files)
        geometry = np.fromiter(map(self.__transform, textio), dtype=object, count=len(files))
        for obj in textio:
            obj.close()
        return geometry


construct = CallableConstructPoly()


class LazyGeoLoc(_LocIndexer):

    def __getitem__(self, item):
        result = super().__getitem__(item)
        if result is None and item[1] == 'geometry':
            raise NotImplementedError()
        elif isinstance(result, (GeoRegions, GeoSeries)):
            isna = result.geometry.isna()
            if count := isna.sum():
                loc = isna.index[isna]
                poly = super().__getitem__((loc, 'poly'))
                geometry: np.ndarray = np.fromiter(construct(poly), dtype=object, count=count)
                self.__setitem__((loc, 'geometry'), geometry)
        return super(LazyGeoLoc, self).__getitem__(item)


class GeoRegions(GeoDataFrame):

    @property
    def loc(self, *args, **kwargs) -> LazyGeoLoc:
        return LazyGeoLoc('loc', self)

    @property
    def _constructor(self):
        return GeoRegions


class Suggest:
    def __new__(cls, *args, **kwargs):
        if not hasattr(cls, '_instance'):
            cls._instance = super().__new__(cls)
        return cls._instance

    @cached_property
    def _continents(self) -> GeoRegions:
        _continents: list[Africa] = [
            getattr(pyrosm.data.sources, continent)
            for continent in 'africa antarctica asia australia_oceania europe north_america south_america'.split()
        ]

        pbf = np.array([
            continent.continent['url']
            for continent in _continents
        ])
        poly = np.char.add(
            np.char.rpartition(pbf, '-latest')[:, 0],
            '.poly'
        )
        continents = np.array([
            continent.continent['name'].rpartition('-latest')[0]
            for continent in _continents
        ], dtype=str, )
        index = pd.Index(continents, name='continent')

        return GeoRegions({
            'pbf': pbf,
            'poly': poly,
            'geometry': None,
        }, index=index, crs=4326)

    @cached_property
    def _regions(self) -> GeoRegions:
        # TODO: There is actually a few seconds delay downloading all the polygons
        _continents: list[Africa] = [
            getattr(pyrosm.data.sources, continent)
            for continent in 'africa antarctica asia australia_oceania europe north_america south_america'.split()
        ]

        continents = np.array([
            continent.continent['name'].rpartition('-latest')[0]
            for continent in _continents
        ], dtype=str, )
        repeat = np.fromiter((
            len(continent.regions)
            for continent in _continents
        ), dtype=int, count=len(_continents))
        continents = continents.repeat(repeat)

        pbf = np.array([
            source['url']
            for continent in _continents
            for source in continent._sources.values()
        ], dtype=str, )
        poly = np.char.add(
            np.char.rpartition(pbf, '-latest')[:, 0],
            '.poly'
        )
        region = np.array([
            source['name'].rpartition('-latest')[0]
            for continent in _continents
            for source in continent._sources.values()
        ], dtype=str, )

        index = pd.MultiIndex.from_arrays([continents, region], names=['continent', 'region'])
        result = GeoRegions({
            'pbf': pbf,
            'poly': poly,
            'geometry': None,
        }, index=index, crs=4326)

        # TODO: Perhaps this could be dynamically determined from geofabrik rather than hardcoded
        result = result.drop('''
        south-africa-and-lesotho
        us-midwest
        us-northeast
        us-pacific
        us-south
        us-west
        alps
        britain-and-ireland
        germany-austria-and-switzerland
        '''.split(), errors='ignore', level='region')

        return result

    @cached_property
    def _subregions(self) -> GeoRegions:
        s = pyrosm.data.sources.subregions
        _countries: list[Brazil] = [
            getattr(s, region)
            for region in s.regions
        ]
        repeat = np.fromiter((
            len(subregion._sources)
            for subregion in _countries
        ), dtype=int, count=len(_countries))

        countries = np.array([
            str.rpartition(country.country['name'], '-latest')[0]
            for country in _countries
        ], dtype=str, )
        countries = countries.repeat(repeat)
        regions = np.array([
            str.rpartition(subregion['name'], '-latest')[0]
            for country in _countries
            for subregion in country._sources.values()
        ], dtype=str, )
        index = pd.MultiIndex.from_arrays([countries, regions], names=['country', 'subregion'])

        pbf = np.array([
            subregion['url']
            for country in _countries
            for subregion in country._sources.values()
        ], dtype=str, )
        poly = np.char.add(
            np.char.rpartition(pbf, '-latest')[:, 0],
            '.poly'
        )

        return GeoRegions({
            'pbf': pbf,
            'poly': poly,
            'geometry': None,
        }, index=index, crs=4326)

    @cached_property
    def _cities(self) -> GeoRegions:
        c = pyrosm.data.sources.cities
        cities = np.array(list(c._sources.keys()), dtype=str, )
        index = pd.Index(cities, name='city')
        pbf = np.array([
            city['url']
            for city in c._sources.values()
        ], dtype=str, )
        poly = np.array([
            str.rpartition(city['url'], '.osm.pbf')[0] + '.poly'
            for city in c._sources.values()
        ], dtype=str, )
        return GeoRegions({
            'pbf': pbf,
            'poly': poly,
            'geometry': None,
        }, index=index, crs=4326)

    def cities(self, bbox: BaseGeometry, *args, url=False) -> np.array:
        # It does not appear that there is a way to subquery cities
        cities = self._cities.loc[:]
        cities = cities[cities.intersects(bbox)]
        if url:
            return cities['pbf'].values
        else:
            return cities.index.get_level_values('city').values

    def regions(self, bbox: BaseGeometry, *args, url=False) -> np.array:
        continents: np.ndarray = self.continents(bbox)
        regions: GeoRegions = self._regions.loc[idx[continents, :], :]
        regions = regions[regions.intersects(bbox)]
        if url:
            return regions['pbf'].values
        else:
            return regions.index.get_level_values('region').values

    def subregions(self, bbox: BaseGeometry, *args, url=False) -> np.array:
        regions: np.ndarray = self.regions(bbox)
        subregions: GeoRegions = self._subregions.loc[idx[regions, :], :]
        subregions = subregions[subregions.intersects(bbox)]
        if url:
            return subregions['pbf'].values
        else:
            return subregions.index.get_level_values('subregion').values

    def continents(self, bbox: BaseGeometry, *args, url=False) -> np.array:
        continents = self._continents.loc[:]
        continents = continents[continents.geometry.intersects(bbox)]
        if url:
            return continents['pbf'].values
        else:
            return continents.index.get_level_values('continent').values


suggest = Suggest()

if __name__ == '__main__':
    chicago = shapely.geometry.box(-87.629, 41.878, -87.614, 41.902)
    montreal = shapely.geometry.box(-73.5, 45.5, -73.4, 45.6)
    sao_paulo = shapely.geometry.box(-46.5, -23.5, -46.4, -23.4)
    cape_town = shapely.geometry.box(-33.5, -18.5, -33.4, -18.4)
    timbuktu = shapely.geometry.box(16.5, -4.5, 16.6, -4.4)
    belize = shapely.geometry.box(-90.5, 17.5, -90.4, 17.6)
    abuja = shapely.geometry.box(7.5, 9.5, 7.6, 9.6)
    nairobi = shapely.geometry.box(-36.5, -1.5, -36.4, -1.4)

    suggest.continents(chicago)
    suggest.regions(chicago)
    suggest.subregions(chicago)
