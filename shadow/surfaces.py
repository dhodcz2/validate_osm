import numpy
from geopandas import GeoDataFrame, GeoSeries
from pandas import Series, DataFrame

import concurrent.futures
import concurrent.futures
import warnings
from typing import Type, Union, Optional
from weakref import WeakKeyDictionary
import dask.dataframe as dd
import dask_geopandas

import dask_geopandas as dgpd
import numpy as np
import pygeos
from geopandas import GeoDataFrame
from geopandas import GeoSeries
import rasterstats
# from util import  get_utm_from_lon_lat, get_raster_path, get_shadow_image
from .util import get_utm_from_lon_lat, get_raster_path, get_shadow_image

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
        source: Union[str, list[str]],
        osmium_executable_path: str = None,
        bbox: Optional[list[float, ...]] = None,
        bbox_latlon=True
):
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

    @property
    def raster(self) -> str:
        return get_raster_path( *self.bbox, self._instance._instance._shadow_dir )

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
        self.cache: WeakKeyDictionary[Surfaces, GeoDataFrame] = WeakKeyDictionary()

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

        self.gdf[id] = self.wkbfab.create_multipolygon(a)
        if 'name' in tags:
            self.name[id] = tags['name']

    def apply_file(self, filename, locations=False, idx='flex_mem'):
        for item in self.__dict__:
            if isinstance(item, dict):
                item.clear()
        super(DescriptorParks, self).apply_file(filename, locations, idx)

    def __get__(self, instance: 'Surfaces', owner: Type['_Footprints']) -> GeoDataFrame:
        if instance not in self.cache:
            self.apply_file(instance._file, locations=True)
            self.cache[instance] = self._get_gdf()
        return self

    def __set__(self, instance, value):
        self.cache[instance] = value

    def __delete__(self, instance):
        del self.cache[instance]

    def _get_gdf(self) -> GeoDataFrame:
        index = np.fromiter(self.gdf.keys(), dtype=np.uint64, count=len(self.gdf))
        geometry = GeoSeries.from_wkb(list(self.gdf.values()), index=index)

        index = np.fromiter(self.name.keys(), dtype=np.uint64, count=len(self.name))
        name = np.fromiter(self.name.values(), dtype='U128', count=len(self.name))
        name = Series(name, index=index)

        index = np.fromiter(self.ways, dtype=np.uint64, count=len(self.ways))
        ways = np.full(len(self.ways), True, dtype=bool)
        ways = Series(ways, index=index)

        if np.all(name.index.values == geometry.index.values):
            raise RuntimeError('you can do this better')
        return GeoDataFrame({
            'name': name,
            'way': ways,
        }, crs=4326, geometry=geometry)

    @property
    def gdf(self) -> GeoDataFrame:
        # TODO:
        ...



class DescriptorNetwork:
    def __get__(self, instance: 'DescriptorNetworks', owner: Type['DescriptorNetworks']):
        self._instance = instance
        if instance not in self._cache:
            osm: pyrosm.OSM = instance._osm[instance._instance]
            nodes, geometry = osm.get_network(self._network_type, None, True)
            self._bbox[instance] = nodes.total_bounds

            nodes: GeoDataFrame
            geometry: GeoDataFrame
            geometry = geometry['id geometry u v length surface'.split()]
            lon = nodes['lon'].mean()
            lat = nodes['lat'].mean()
            crs = get_utm_from_lon_lat(lon, lat)
            geometry['geometry'] = (
                GeoSeries.to_crs(geometry['geometry'], crs)
                    .buffer(4)
                    .to_crs(4326)
            )
            self._cache[instance] = nodes, geometry
        return self

    @property
    def bbox(self) -> list[float, ...]:
        return self._bbox[self._instance]

    def __init__(self, network_type: str):
        self._network_type = network_type
        self._cache: WeakKeyDictionary[DescriptorNetworks, tuple[GeoDataFrame, GeoDataFrame]] = WeakKeyDictionary()
        self._bbox: WeakKeyDictionary[DescriptorNetworks, list[float, ...]] = WeakKeyDictionary()

    @property
    def geometry(self) -> GeoDataFrame:
        return self._cache[self._instance][1]

    @geometry.setter
    def geometry(self, value):
        nodes, geometry = self._cache[self._instance]
        self._cache[self._instance] = (nodes, value)

    @geometry.deleter
    def geometry(self):
        del self._cache[self._instance]

    @property
    def nodes(self) -> GeoDataFrame:
        return self._cache[self._instance][0]

    @nodes.setter
    def nodes(self, value):
        nodes, geometry = self._cache[self._instance]
        self._cache[self._instance] = (value, geometry)

    @nodes.deleter
    def nodes(self):
        del self._cache[self._instance]

    @property
    def raster(self) -> str:
        return get_raster_path( *self.bbox, self._instance._instance._shadow_dir )


class DescriptorNetworks:
    walking = DescriptorNetwork('walking')
    cycling = DescriptorNetwork('cycling')
    driving = DescriptorNetwork('driving')
    driving_service = DescriptorNetwork('driving_service')
    all = DescriptorNetwork('all')

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
        self._file = file
        self._shadow_dir = shadow_dir


if __name__ == '__main__':
    path = pyrosm_extract(
        'newyorkcity',
        osmium_executable_path='~/PycharmProjects/StaticOSM/work/osmium-tool/build/osmium',
        bbox=[40.6986519312932, -74.04222185978449, 40.800217630179155, -73.92257387648877],
    )
    surfaces = Surfaces(path)
    surfaces.networks.driving.geometry
    # surfaces.parks
    # surfaces.networks.driving
    # driving = surfaces.networks.driving
    # print()
