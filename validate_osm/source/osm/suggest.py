import concurrent.futures
from pandas import IndexSlice as idx

import functools

import geopandas as gpd

import pygeos
from geopandas import GeoDataFrame, GeoSeries
from pandas import Series, DataFrame

import re
from functools import cached_property
from typing import Iterator, Iterable

import numpy as np
import pandas as pd
import pyrosm
import requests
import shapely.geometry
from geopandas import GeoDataFrame
from pandas.core.indexing import _LocIndexer
from pyrosm.data import Africa
from pyrosm.data import Brazil
from shapely.geometry import MultiPolygon, Polygon
from shapely.geometry.base import BaseGeometry


def __construct(text: str) -> MultiPolygon:
    text = text.replace('\n', '')
    matches = re.findall(r'(?<=\d)(.*?)END', text)
    if not matches:
        raise ValueError('No polygons found')
    splits = list(map(str.split, matches))
    lons = (np.array(split[::2], dtype=np.float64) for split in splits)
    lats = (np.array(split[1::2], dtype=np.float64) for split in splits)
    polygons = map(Polygon, map(zip, lons, lats))
    multipolygon = MultiPolygon(polygons)
    return multipolygon


def construct(urls: Iterable[str]) -> Iterator[MultiPolygon]:
    session = requests.Session()
    threads = concurrent.futures.ThreadPoolExecutor()
    for response in threads.map(session.get, urls):
        response.raise_for_status()
        yield __construct(response.text)


class LookUpRegions:
    def __new__(cls, *args, **kwargs):
        # Cache the structure for multiple suggest calls without instantiating everything at import time
        if not hasattr(cls, '_instance'):
            cls._instance = super().__new__(cls)
        return cls._instance

    @cached_property
    def continents(self) -> DataFrame:
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
            continent.continent['name']
            for continent in _continents
        ], dtype=str, )
        index = pd.Index(continents, name='continent')

        return DataFrame({
            'pbf': pbf,
            'poly': poly,
        }, index=index)

    @cached_property
    def regions(self) -> DataFrame:
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
        return DataFrame({
            'pbf': pbf,
            'poly': poly,
        }, index=index)

    @cached_property
    def subregions(self) -> DataFrame:
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

        return DataFrame({
            'pbf': pbf,
            'poly': poly,
        }, index=index)

    @cached_property
    def cities(self) -> DataFrame:
        c = pyrosm.data.sources.cities
        cities = np.array(list(c._sources.keys()), dtype=str, )
        pbf = np.array([
            city['url']
            for city in c._sources.values()
        ], dtype=str, )
        poly = np.array([
            str.rpartition(city['url'], '.osm.pbf')[0] + '.poly'
            for city in c._sources.values()
        ], dtype=str, )
        return DataFrame({
            'pbf': pbf,
            'poly': poly,
        }, index=cities)


class LookUpLocIndexer(_LocIndexer):
    def __getitem__(self, item):
        obj: GeoRegions = self.obj
        binding = obj.binding
        if (
                isinstance(item, slice)
                and item.step is None
                and item.start is None
                and item.stop is None
        ):
            index = binding.index
        else:
            index: pd.Index = binding.loc[item].index

        if dif := index.difference(obj.index):
            poly = binding.loc[dif, 'poly']
            obj.loc[dif, 'geometry'] = np.fromiter(construct(poly), dtype=object)
            obj.loc[dif, 'pbf'] = binding.loc[dif, 'pbf']

        return super(LookUpLocIndexer, self).__getitem__(item)


class GeoRegions(GeoDataFrame):
    _metadata = GeoDataFrame._metadata + ['binding']

    def __init__(self, binding: DataFrame, *args, **kwargs, ):
        kwargs['columns'] = ['pbf', 'geometry']
        kwargs['crs'] = 4326
        super().__init__(*args, **kwargs)
        index = binding.index
        self.binding = binding
        # self.index = index.__class__([], names=index.names)
        if isinstance(index, pd.MultiIndex):
            self.index = pd.MultiIndex(levels=[[], []], codes=[[], []], names=index.names)
        elif isinstance(index, pd.Index):
            self.index = pd.Index([], name=index.name)

    def loc(self) -> LookUpLocIndexer:
        return LookUpLocIndexer(self)

    # @cached_property
    # def loc(self) -> _LocIndexer:
    #     # Automatically load and cache the geometries required by the key
    #     loc = super(GeoRegions, self).loc()
    #
    #     def __getitem__(self_, key):
    #         print('getitem')
    #         binding = self.binding
    #         if (
    #                 isinstance(key, slice)
    #                 and key.start is None
    #                 and key.stop is None
    #                 and key.step is None
    #         ):
    #             index = binding.index
    #         else:
    #             index: pd.Index = binding.loc[key].index
    #
    #         if dif := index.difference(self.index):
    #             poly = binding.loc[dif, 'poly']
    #             self.loc[dif, 'geometry'] = np.fromiter(construct(poly), dtype=object)
    #             self.loc[dif, 'pbf'] = binding.loc[index, 'pbf']
    #
    #         return super(_LocIndexer, self_).__getitem__(key)
    #
    #     loc.__getitem__ = __getitem__
    #     return loc
    #


class Suggest:
    def __init__(self):
        lookup = LookUpRegions()
        self._continents = GeoRegions(lookup.continents)
        self._regions = GeoRegions(lookup.regions)
        self._subregions = GeoRegions(lookup.subregions)
        self._cities = GeoRegions(lookup.cities)

    def cities(self, bbox: BaseGeometry, *args, url=False) -> np.array:
        # It does not appear that there is a way to subquery cities
        loc = self._cities.loc
        cities = loc[:]
        # cities = self._cities.loc[:]
        cities = cities[cities.intersects(bbox)]
        if url:
            return cities['pbf'].values
        else:
            return cities.index.get_level_values('city').values

    def regions(self, bbox: BaseGeometry, *args, url=False) -> np.array:
        continents: np.ndarray = self.continents(bbox)
        regions: GeoDataFrame = self._regions.loc[idx[continents, :], :]
        regions = regions[regions.intersects(bbox)]
        if url:
            return regions['pbf'].values
        else:
            return regions.index.get_level_values('region').values

    def subregions(self, bbox: BaseGeometry, *args, url=False) -> np.array:
        regions: np.ndarray = self.regions(bbox)
        subregions: GeoDataFrame = self._subregions.loc[idx[regions, :], :]
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

'''
suggest.cities()
suggest.regions()
suggest.continents()
'''

if __name__ == '__main__':
    lookup = LookUpRegions()
    lookup.continents
    lookup.regions
    lookup.cities
    lookup.subregions

    chicago = shapely.geometry.box(41.878, -87.629, 41.902, -87.614)
    manhattan = shapely.geometry.box(40.878, -73.629, 40.902, -73.614)
    moscow = shapely.geometry.box(55.878, 37.629, 55.902, 37.614)
    kyiv = shapely.geometry.box(50.878, 30.629, 50.902, 30.614)
    berlin = shapely.geometry.box(52.878, 13.629, 52.902, 13.614)
    mogadishu = shapely.geometry.box(1.878, 45.629, 1.902, 45.614)

    suggest.continents(chicago)
    suggest.regions(chicago)
    suggest.subregions(chicago)
    suggest.cities(chicago)
    print()
