import multiprocessing
import argparse
import concurrent.futures
import concurrent.futures
import math
import multiprocessing
import time
import warnings
from typing import Type, Union, Optional
from weakref import WeakKeyDictionary

import numpy as np
import rasterstats
import shapely.geometry.base
from geopandas import GeoDataFrame
from geopandas import GeoSeries
from pandas import Series, DataFrame

from shadow.util import get_utm_from_lon_lat, get_raster_path

warnings.filterwarnings('ignore', '.*PyGEOS.*')

import osmium

import pandas as pd

import functools
import itertools
import os
import tempfile
from typing import Iterator

import pyrosm


def osmium_extract(
        file: str,
        osmium_executable_path: str,
        bbox: list[float, ...],
        bbox_latlon=True
) -> str:
    if bbox_latlon:
        bbox = [bbox[1], bbox[0], bbox[3], bbox[2]]
    string: str = ','.join(str(b) for b in bbox)
    name = file.rpartition('/')[2] if '/' in file else file
    tempdir = tempfile.gettempdir()
    if not os.path.exists(tempdir):
        os.makedirs(tempdir)
    temp = os.path.join(tempdir, name)
    os.system(f'{osmium_executable_path} extract -b {string} {file} -o {temp} --overwrite')
    return temp


@functools.singledispatch
def pyrosm_extract(
        source: str,
        osmium_executable_path: str = None,
        bbox: Optional[list[float, ...]] = None,
        bbox_latlon=True
) -> str:
    path = pyrosm.get_data(source)
    if bbox:
        path = osmium_extract(path, osmium_executable_path, bbox, bbox_latlon)
    return path


@pyrosm_extract.register(list)
def _(
        source: list[str],
        osmium_executable_path: str = None,
        bbox: Union[list[float, ...], None] = None,
        bbox_latlon=True
) -> Iterator[str]:
    with concurrent.futures.ThreadPoolExecutor() as threads, concurrent.futures.ProcessPoolExecutor() as processes:
        files = threads.map(pyrosm.get_data, source)
        if bbox is not None:
            yield from processes.map(
                osmium_extract,
                itertools.repeat(osmium_executable_path, len(source)),
                bbox,
                itertools.repeat(bbox_latlon)
            )
        else:
            yield from files


class DescriptorParks(osmium.SimpleHandler):
    wkbfab = osmium.geom.WKBFactory()

    def __init__(self):
        super(DescriptorParks, self).__init__()
        self.natural = {'wood', 'grass'}
        self.land_use = {
            'wood', 'grass', 'forest', 'orchard', 'village_green',
            'vineyard', 'cemetery', 'meadow', 'village_green'
        }
        self.leisure = {
            'dog_park', 'park', 'playground', 'recreation_ground',
        }
        self.name = {}
        self.geometry = {}
        self.ways = set()
        self._cache: WeakKeyDictionary[Surfaces, GeoDataFrame] = WeakKeyDictionary()
        self._bbox: WeakKeyDictionary[Surfaces, list[float, ...]] = WeakKeyDictionary()

    def area(self, a: osmium.osm.Area):
        # TODO: What about nodes marked 'point of interest'?
        tags: osmium.osm.TagList = a.tags
        # Qualifiers
        if not (
                tags.get('natural', None) in self.natural
                or tags.get('land_use', None) in self.land_use
                or tags.get('leisure', None) in self.leisure
        ):
            return

        id = a.orig_id()
        if a.from_way():
            self.ways.add(id)

        self.geometry[id] = self.wkbfab.create_multipolygon(a)
        if 'name' in tags:
            self.name[id] = tags['name']

    def apply_file(self, filename, locations=False, idx='flex_mem'):
        for item in self.__dict__:
            if isinstance(item, dict):
                item.clear()
        super(DescriptorParks, self).apply_file(filename, locations, idx)

    def __get__(self, instance: 'Surfaces', owner: Type['_Footprints']) -> 'DescriptorParks':
        self._instance = instance
        return self

    def __set__(self, instance, value):
        self._cache[instance] = value

    def __delete__(self, instance):
        del self._cache[instance]

    @property
    def gdf(self) -> GeoDataFrame:
        instance = self._instance
        if instance not in self._cache:
            self.apply_file(instance._file, locations=True)
            index = np.fromiter(self.geometry.keys(), dtype=np.uint64, count=len(self.geometry))
            geometry = GeoSeries.from_wkb(list(self.geometry.values()), index=index)

            index = np.fromiter(self.name.keys(), dtype=np.uint64, count=len(self.name))
            name = np.fromiter(self.name.values(), dtype='U128', count=len(self.name))
            name = Series(name, index=index, dtype='string')

            index = np.fromiter(self.ways, dtype=np.uint64, count=len(self.ways))
            ways = np.full(len(self.ways), True, dtype=bool)
            ways = Series(ways, index=index, dtype='boolean')

            if np.all(name.index.values == geometry.index.values):
                raise RuntimeError('you can do this better')
            gdf = GeoDataFrame({
                'name': name,
                'way': ways,
            }, crs=4326, geometry=geometry)
            gdf.loc[gdf['way'].isna(), 'way'] = False
            self.__set__(instance, gdf)
            return gdf
        return self._cache[instance]

    @gdf.setter
    def gdf(self, value):
        self._cache[self._instance] = value

    @gdf.deleter
    def gdf(self):
        del self._cache[self._instance]

    @staticmethod
    def sum(interface: dict, tiff: str):
        return [
            stats['sum']
            for stats in rasterstats.gen_zonal_stats(interface, tiff, stats='sum')
        ]

    @functools.singledispatchmethod
    @classmethod
    def rasterstats_from_file(cls, file: str, shadow_dir: str, zoom: int) -> GeoDataFrame:
        surfaces = Surfaces(file, shadow_dir)
        parks = surfaces.parks
        gdf = parks.gdf
        raster = get_raster_path(
            *gdf.total_bounds,
            zoom=zoom,
            basedir=shadow_dir,
        )

        t = time.time()
        step = math.ceil(len(gdf) / multiprocessing.cpu_count())
        slices = (
            slice(l, l + step)
            for l in range(0, len(gdf), step)
        )
        interfaces = [
            getattr(gdf.iloc[slice], '__geo_interface__')
            for slice in slices
        ]
        with concurrent.futures.ProcessPoolExecutor() as processes:
            substats = processes.map(cls.sum, interfaces, itertools.repeat(raster))
        sum = np.fromiter(itertools.chain.from_iterable(substats), dtype=np.float64, count=len(gdf))
        print(f'{time.time()-t} for parallel')
        t = time.time()
        _sum = np.fromiter((
            stats['sum']
            for stats in rasterstats.gen_zonal_stats(gdf, raster, stats='sum')
        ), dtype=np.float64, count=len(gdf))
        print(f'{t-time.time()} for serial')

        result = GeoDataFrame({
            'name': gdf['name'].values,
            'sum': sum,
        }, index=gdf.index, crs=gdf.crs, geometry=gdf['geometry'].values)
        result['sum'] = pd.Series.astype(result['sum'], 'UInt32')
        return result

    @rasterstats_from_file.register(list)
    @classmethod
    def _(cls, files: list[str], shadow_dir: str, zoom: int) -> DataFrame:
        it_stats = concurrent.futures.ProcessPoolExecutor().map(
            DescriptorParks.rasterstats_from_file,
            files,
            itertools.repeat(shadow_dir),
            itertools.repeat(zoom),
        )
        sources = (
            file.rpartition('/')[2]
            for file in files
            if '/' in file
        )
        sources = (
            source.rpartition('.')[0]
            for source in sources
        )
        it_stats = (
            stats.assign(name=name)
            for stats, name in zip(it_stats, sources)
        )
        result = pd.concat(
            it_stats,
            axis=0,
            names='name area sum'.split(),
            sort=False,
        )
        return result


class DescriptorNetwork:
    network_type: str

    @functools.singledispatchmethod
    @classmethod
    def rasterstats_from_file(cls, file: str, shadow_dir: str, zoom: int) -> DataFrame:
        surfaces = Surfaces(file, shadow_dir)
        network: DescriptorNetwork = surfaces.networks.__getattribute__(cls.network_type)
        gdf = network.gdf
        raster = get_raster_path(
            *gdf.total_bounds,
            zoom=zoom,
            basedir=shadow_dir,
        )
        centroid: shapely.geometry.base.BaseGeometry = gdf.geometry.iloc[0].centroid
        lon = centroid.x
        lat = centroid.y
        crs = get_utm_from_lon_lat(lon, lat)
        geometry = (
            GeoSeries.to_crs(gdf['geometry'], crs)
                .buffer(4)
                .to_crs(4326)
        )
        sum = [
            stats['sum']
            for stats in rasterstats.gen_zonal_stats(geometry, raster, stats='sum')
        ]

        result = GeoDataFrame({
            'name': gdf['name'].values,
            'sum': sum,
            'length': gdf['length'].values,
        }, index=gdf.index, crs=gdf.crs, geometry=gdf['geometry'].values)
        return result

        # df = DataFrame({
        #     'name': gdf['name'],
        #     'area': gdf.area,
        #     'sum': sum,
        # }, index=gdf.index)
        # df['sum'] = pd.Series.astype(df['sum'], 'UInt32')
        #
        # return df
        #

    @rasterstats_from_file.register(list)
    @classmethod
    def _(cls, files: list[str], shadow_dir: str, zoom: int) -> DataFrame:
        it_stats = concurrent.futures.ProcessPoolExecutor().map(
            DescriptorParks.rasterstats_from_file,
            files,
            itertools.repeat(shadow_dir),
            itertools.repeat(zoom),
        )
        sources = (
            file.rpartition('/')[2]
            for file in files
            if '/' in file
        )
        sources = (
            source.rpartition('.')[0]
            for source in sources
        )
        it_stats = (
            stats.assign(name=name)
            for stats, name in zip(it_stats, sources)
        )
        result = pd.concat(
            it_stats,
            axis=0,
            names='name area sum'.split(),
            sort=False,
        )
        return result

    def __get__(self, instance: 'DescriptorNetworks', owner: Type['DescriptorNetworks']):
        self._instance = instance
        # if instance not in self._cache:
        #     self._cache[instance] = self._get_network()
        return self

    def _get_network(self) -> tuple[GeoDataFrame, GeoDataFrame]:
        instance = self._instance
        osm: pyrosm.OSM = instance._osm[instance._instance]
        nodes, geometry = osm.get_network(self._network_type, None, True)
        self._bbox[instance] = nodes.total_bounds

        nodes: GeoDataFrame
        geometry: GeoDataFrame
        geometry = geometry['id geometry u v length surface'.split()]
        return nodes, geometry

    def __init__(self):
        self._cache: WeakKeyDictionary[DescriptorNetworks, tuple[GeoDataFrame, GeoDataFrame]] = WeakKeyDictionary()
        self._bbox: WeakKeyDictionary[DescriptorNetworks, list[float, ...]] = WeakKeyDictionary()

    @property
    def gdf(self) -> GeoDataFrame:
        instance = self._instance
        if instance not in self._cache:
            self._cache = self._get_network()
        return self._cache[instance][1]

    @gdf.setter
    def gdf(self, value):
        nodes, geometry = self._cache[self._instance]
        self._cache[self._instance] = (nodes, value)

    @gdf.deleter
    def gdf(self):
        del self._cache[self._instance]

    @property
    def nodes(self) -> GeoDataFrame:
        instance = self._instance
        if instance not in self._cache:
            self._cache = self._get_network()
        return self._cache[instance][0]

    @nodes.setter
    def nodes(self, value):
        nodes, geometry = self._cache[self._instance]
        self._cache[self._instance] = (value, geometry)

    @nodes.deleter
    def nodes(self):
        del self._cache[self._instance]


class DescriptorWalkingNetwork(DescriptorNetwork):
    network_type = 'walking'


class DescriptorCyclingNetwork(DescriptorNetwork):
    network_type = 'cycling'


class DescriptorDrivingNetwork(DescriptorNetwork):
    network_type = 'driving'


class DescriptorDrivingServiceNetwork(DescriptorNetwork):
    network_type = 'driving_service'


class DescriptorAllNetwork(DescriptorNetwork):
    network_type = 'all'


class DescriptorNetworks:
    walking = DescriptorWalkingNetwork()
    cycling = DescriptorCyclingNetwork()
    driving = DescriptorDrivingNetwork()
    driving_service = DescriptorDrivingServiceNetwork()
    all = DescriptorAllNetwork()

    def __init__(self):
        self._instance: Optional['Surfaces'] = None
        self._osm: WeakKeyDictionary[Surfaces, pyrosm.OSM] = WeakKeyDictionary()

    def __get__(self, instance: 'Surfaces', owner):
        self._instance = instance
        if instance not in self._osm:
            self._osm[instance] = pyrosm.OSM(self._instance._file)
            # TODO: Use bounding box, generate raster
        return self


class Surfaces:
    parks = DescriptorParks()
    networks = DescriptorNetworks()

    def __init__(self, file: str, shadow_dir: str):
        if file.rpartition('.')[2] != 'pbf':
            raise ValueError(f"{file=} is not a PBF file")
        self._file = file
        self._shadow_dir = shadow_dir


class _Namespace:
    input: list[str]
    output: str
    shadows: str
    verbose: bool
    zoom: int
    osmium: str
    parks: str
    driving: str
    all: str
    driving_service: str
    walking: str
    cycling: str


# if __name__ == '__main__':
#     parser = argparse.ArgumentParser(description='generate elevation map from geometric data')
#     parser.add_argument(
#         'input',
#         nargs='*',
#         help='input files, could be a pyrosm string or .pbf file',
#     )
#     parser.add_argument(
#         'shadows',
#         type=str,
#         help='directory from which the shadow raster may be generated from png files',
#         default=''
#     )
#     parser.add_argument(
#         '--verbose',
#         '-v',
#         type=bool,
#         action='store_true',
#     )
#     parser.add_argument(
#         '--zoom',
#         '-z',
#         nargs='+',
#         type=int,
#         help='slippy tile zoom level'
#     )
#     parser.add_argument(
#         '--parks',
#         type=str,
#     )
#     parser.add_argument(
#         '--driving',
#         type=str
#     )
#     parser.add_argument(
#         '--all',
#         type=str,
#     )
#     parser.add_argument(
#         '--driving_service',
#         type=str
#     )
#     parser.add_argument(
#         '--cycling',
#         type=str,
#     )
#     parser.add_argument(
#         '--walking',
#         type=str,
#     )
#
#     args = parser.parse_args(namespace=_Namespace)
#     files = [
#         input
#         for input in args.input
#         if '.' in input
#     ]
#     sources = [
#         input
#         for input in args.input
#         if '.' not in input
#     ]
#     if sources:
#         with concurrent.futures.ThreadPoolExecutor() as threads:
#             temps = threads.map(pyrosm.get_data, sources)
#             files.extend(temps)
#     with concurrent.futures.ProcessPoolExecutor() as processes:
#         gdfs = processes.map()

if __name__ == '__main__':
    path = pyrosm_extract(
        'newyorkcity',
        osmium_executable_path='~/PycharmProjects/StaticOSM/work/osmium-tool/build/osmium',
        bbox=[40.6986519312932, -74.04222185978449, 40.800217630179155, -73.92257387648877],
    )
    stats = Surfaces.parks.rasterstats_from_file(path, '/home/arstneio/Downloads/shadows/', zoom=16)
    stats = Surfaces.networks.driving.rasterstats_from_file(path, '/home/arstneio/Downloads/shadows/', zoom=16)
