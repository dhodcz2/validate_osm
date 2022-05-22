import concurrent.futures
import os
import tempfile
import pathlib
import warnings
import functools
import itertools
import warnings
from typing import Type, Iterator
from weakref import WeakKeyDictionary

import numpy as np
import pyrosm
from geopandas import GeoDataFrame
from geopandas import GeoSeries

warnings.filterwarnings('ignore', '.*PyGEOS.*')

import osmium

import pandas as pd


class ShadowHandler(osmium.SimpleHandler):
    wkbfab = osmium.geom.WKBFactory()

    def gdf(self) -> GeoDataFrame:
        columns: dict = self.__dict__
        geometry = columns.pop('geometry')
        geometry = GeoSeries.from_wkb(list(geometry.values()), index=geometry.keys(), crs=4326)
        series = (
            pd.Series(items.values(), index=items.keys(), name=name, dtype=str)
            for name, items in columns.items()
            if isinstance(items, dict)
        )
        return GeoDataFrame(series, crs=4326, geometry=geometry)

    def apply_file(self, filename, locations=False, idx='flex_mem'):
        for item in self.__dict__:
            if isinstance(item, dict):
                item.clear()
        super(ShadowHandler, self).apply_file(filename, locations, idx)

    def __get__(self, instance: 'Footprints' | None, owner: Type['Footprints']) -> GeoDataFrame:
        if instance not in self.cache:
            self.apply_file(instance.file, locations=True)
            self.cache[instance] = self.gdf()
        return self.cache[instance]

    def __init__(self):
        super(ShadowHandler, self).__init__()
        self.geometry = {}
        self.cache: WeakKeyDictionary[
            Footprints, GeoDataFrame
        ] = WeakKeyDictionary()


class ParkHandler(ShadowHandler):

    def __init__(self):
        super(ParkHandler, self).__init__()
        self.natural = {'wood', 'grass'}
        self.name = {}
        self.land_use = {
            'wood', 'grass', 'forest', 'orchard', 'village_green',
            'vineyard', 'cemetery', 'meadow', 'village_green'
        }
        self.leisure = {
            'dog_park', 'park', 'playground', 'recreation_ground',
        }

    def area(self, a: osmium.osm.Area):
        tags: osmium.osm.TagList = a.tags
        # Qualifiers
        if not (
                tags.get('natural', None) in self.natural
                or tags.get('land_use', None) in self.land_use
                or tags.get('leisure', None) in self.leisure
        ):
            return

        id = f'w{a.orig_id()}' if a.from_way() else f'r{a.orig_id()}'
        self.geometry[id] = self.wkbfab.create_multipolygon(a)
        if 'name' in tags:
            self.name[id] = tags['name']


class FootWayHandler(ShadowHandler):

    def __init__(self):
        super(FootWayHandler, self).__init__()
        self.surface: dict[str, str] = {}
        self.footway: dict[str, str] = {}

    def area(self, a: osmium.osm.Area):
        tags: osmium.osm.TagList = a.tags
        # Qualifiers
        if not (
                'footway' in tags
        ):
            return
        tags: osmium.osm.TagList = a.tags
        id = f'w{a.orig_id()}' if a.from_way() else f'r{a.orig_id()}'
        self.geometry[id] = self.wkbfab.create_multipolygon(a)
        if 'surface' in tags:
            self.surface[id] = tags['surface']
        self.footway[id] = tags['footway']


class HighWayHandler(ShadowHandler):

    def area(self, a: osmium.osm.Area):
        tags: osmium.osm.TagList = a.tags
        # Qualifiers
        if not (
                'highway' in tags
        ):
            return
        id = f'w{a.orig_id()}' if a.from_way() else f'r{a.orig_id()}'
        self.geometry[id] = self.wkbfab.create_multipolygon(a)
        if 'surface' in tags:
            self.surface[id] = tags['surface']
        self.highway[id] = tags['highway']

    def __init__(self):
        self.surface: dict[str, str] = {}
        self.highway: dict[str, str] = {}
        super(HighWayHandler, self).__init__()


class Footprints:
    def __init__(self, file):
        self.file = file
        self.parks = ParkHandler()
        self.sidewalks = FootWayHandler()
        self.highways = HighWayHandler()

    @functools.cached_property
    def geometry(self) -> GeoSeries:
        parks = self.parks.geometry
        sidewalks = self.sidewalks.geometry
        highways = self.highways.geometry
        index = pd.MultiIndex.from_arrays((
            np.concatenate((
                parks.index.values,
                sidewalks.index.values,
                highways.index.values,
            )),
            np.concatenate((
                np.repeat('park', len(parks)),
                np.repeat('sidewalk', len(sidewalks)),
                np.repeat('highway', len(highways)),
            ), dtype='S8')
        ), names=['id', 'type'])
        data = itertools.chain(parks, sidewalks, highways)
        return GeoSeries(data, index=index)


@functools.singledispatch
def get_footprints(
        source: str,
        bbox: None | list[float] = None,
        osmium_tool_program_path: None | str = None,
        bbox_latlon=True,
) -> Footprints:
    if '.' not in source:
        path = pyrosm.get_data(source)
    else:
        path = source

    if bbox is not None:
        if '/' in path:
            filename = path.rpartition('/')[2]
        else:
            filename = path
        if bbox_latlon:
            bbox = [bbox[1], bbox[0], bbox[3], bbox[2]]

        temp = pathlib.Path(tempfile.gettempdir()) / filename
        string = ','.join(str(coord) for coord in bbox)
        os.system(f"{osmium_tool_program_path} extract -b {string} {path} -o {temp} --overwrite")
        footprints = Footprints(temp)
    else:
        footprints = Footprints(path)
    return footprints


@get_footprints.register
def _(
        source: list[str],
        bbox: None | list[float] = None,
        osmium_tool_program_path: None | str = None,
        bbox_latlon=True,
) -> Iterator[Footprints]:
    with concurrent.futures.ThreadPoolExecutor() as threads, concurrent.futures.ProcessPoolExecutor() as processes:
        datasets = (
            s
            for s in source
            if '.' not in s
        )
        temps = threads.map(pyrosm.get_data, datasets).__iter__()
        sources: Iterator[str] = (
            s
            if '.' in s
            else next(temps)
            for s in source
        )
        fn = functools.partial(
            get_footprints,
            bbox=bbox,
            osmium_tool_program_path=osmium_tool_program_path,
            bbox_latlon=bbox_latlon,
        )
        yield from processes.map(fn, sources)


def load_footprints(
        source: str,
        destination: None | str = None,
        bbox: None | list[float] = None,
        osmium_tool_program_path: None | str = None,
        bbox_latlon=True,
) -> None:
    ...


if __name__ == '__main__':
    ...
