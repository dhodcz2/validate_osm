import concurrent.futures
import shapely.geometry
import re
from functools import cached_property
from typing import Iterator, Iterable

import numpy as np
import pandas as pd
import pyrosm
import requests
from geopandas import GeoDataFrame
from pandas import IndexSlice as idx
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
        try:
            response.raise_for_status()
        except requests.exceptions.HTTPError:
            yield None
        else:
            yield __construct(response.text)


class LazyLoc(_LocIndexer):
    def __getitem__(self, item):
        obj: GeoDataFrame = super().__getitem__(item)
        loc = obj.geometry.isna()
        if count := loc.sum():
            loc = loc[loc]
            geometry =np.fromiter(construct(obj.loc[loc.index, 'poly']), dtype=object, count=count)
            self.obj.loc[loc.index, 'geometry'] = geometry
        return super(LazyLoc, self).__getitem__(item)


class GeoRegions(GeoDataFrame):

    @property
    def loc(self, *args, **kwargs) -> LazyLoc:
        return LazyLoc('loc', self)

    def __init__(self, *args, **kwargs):
        kwargs['geometry'] = np.full((len(kwargs['index'], )), None)
        super().__init__(*args, **kwargs)


class Suggest:
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
        }, index=index)

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
        return GeoRegions({
            'pbf': pbf,
            'poly': poly,
        }, index=index)

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
        }, index=index)

    @cached_property
    def _cities(self) -> GeoRegions:
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
        return GeoRegions({
            'pbf': pbf,
            'poly': poly,
        }, index=cities)

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

# TODO: Unfortunately there is nothing in thi pyrosm library to indicate a discrepancy
#   between subregions and special sub regions which overlap
#   so we must do some web scraping
if __name__ == '__main__':
    # chicago = shapely.geometry.box(41.878, -87.629, 41.902, -87.614)
    # manhattan = shapely.geometry.box(40.878, -73.629, 40.902, -73.614)
    # moscow = shapely.geometry.box(55.878, 37.629, 55.902, 37.614)
    # kyiv = shapely.geometry.box(50.878, 30.629, 50.902, 30.614)
    # berlin = shapely.geometry.box(52.878, 13.629, 52.902, 13.614)
    # mogadishu = shapely.geometry.box(1.878, 45.629, 1.902, 45.614)

    chicago = shapely.geometry.box(-87.629, 41.878, -87.614, 41.902)
    manhattan = shapely.geometry.box(-73.629, 40.878, -73.614, 40.902)
    moscow = shapely.geometry.box(37.629, 55.878, 37.614, 55.902)
    kyiv = shapely.geometry.box(30.629, 50.878, 30.614, 50.902)
    berlin = shapely.geometry.box(13.629, 52.878, 13.614, 52.902)

    suggest.continents(chicago)
    suggest.regions(chicago)
    suggest.subregions(chicago)
    suggest.cities(chicago)
    print()
