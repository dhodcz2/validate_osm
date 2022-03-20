import dataclasses
import logging

import os
from pathlib import Path
from typing import Generator, Union, Type, Any
from typing import Iterator, Optional
from weakref import WeakKeyDictionary

import geopandas as gpd
import numpy as np
from geopandas import GeoDataFrame

from validate_osm.source.footprint import CallableFootprint
from validate_osm.source.source import Source
from validate_osm.util import concat
from validate_osm.util.scripts import logged_subprocess


@dataclasses.dataclass
class StructData:
    name: str
    dtype: str
    abstract: bool
    crs: Any
    dependent: set[str]

    def __hash__(self):
        return hash(self.name)

    def __eq__(self, other):
        return self.name == other

    def __repr__(self):
        return self.name


#
#
# class DescriptorData:
#     cache: WeakKeyDictionary[object, GeoDataFrame] = WeakKeyDictionary()
#
#     def __init__(self):
#         self._instance = None
#         self._owner = None
#
#     def __get__(self, instance: object, owner: Type) -> Union[GeoDataFrame, 'DescriptorData']:
#         from validate_osm.compare.compare import Compare
#         instance: Optional[Compare]
#         owner: Optional[Type[Compare]]
#         self._instance = instance
#         self._owner = owner
#         if instance is None:
#             return self
#         if instance in self.cache:
#             return self.cache[instance]
#         path = self.path
#         # TODO: Perhaps if there is a file with a bbox that contains instance.bbox, load
#         #   that bbox, .within it,
#         if path.exists() and not instance.ignore_file:
#             with logged_subprocess(
#                     instance.logger,
#                     f'reading {owner.__name__}.data from {path} ({os.path.getsize(path) / 1024 / 1024 :.1f} MB)'
#             ):
#                 data = self.cache[instance] = gpd.read_feather(path)
#         else:
#             with logged_subprocess(instance.logger, f'building {owner.__name__}.data'):
#                 data = self.cache[instance] = self._data()
#
#             if not path.parent.exists():
#                 os.makedirs(path.parent)
#             with logged_subprocess(instance.logger, f'serializing {owner.__name__}.data {path}'):
#                 data.to_feather(path)
#         return data
#
#     def __enter__(self):
#         ...
#
#     def __exit__(self, exc_type, exc_val, exc_tb):
#         ...
#
#     def _data(self) -> GeoDataFrame:
#         from validate_osm.compare.compare import Compare
#         self._instance: Compare
#         self._owner: Type[Compare]
#
#         # We must concatenate DataFrames instead of going purely by Series because some columns
#         #   may return a single-value, which must be repeated to the same length of other columns
#         # To preserve SourceOSM.relation and SourceOSM.way indices, we must first
#         def datas() -> Generator[gpd.GeoDataFrame, None, None]:
#             for name, source in self._instance.sources.items():
#                 data: GeoDataFrame = source.data
#                 data = data.reset_index()
#                 data['name'] = name
#                 if 'id' not in data:
#                     data['id'] = np.nan
#                 if 'group' not in data:
#                     data['group'] = np.nan
#                 data = data.set_index(['name', 'id', 'group'], drop=True)
#                 yield data
#                 self._instance.logger.debug(f'deleting {source.resource.__class__.__name__}')
#                 del source.data
#
#         with logged_subprocess(self._instance.logger, f'concatenating {self.__class__.__name__}.data', ):
#             data = concat(datas())
#
#         # TODO: Investigate how these invalid geometries came to be
#         inval = data[data['geometry'].isna() | data['centroid'].isna()].index
#         if len(inval):
#             self._instance.logger.warning(f'no geom: {inval}')
#             data = data.drop(index=inval)
#
#         self.cache[self._instance] = data
#
#         # with logged_subprocess(self._instance.logger, 'applying footprints'):
#         #     data = footprint(data)
#
#         data = data.sort_index(axis=0)
#         data['iloc'] = range(len(data))
#         data['geometry'] = data['geometry'].to_crs(3857)
#         data['centroid'] = data['centroid'].to_crs(3857)
#         return data
#
#     def __delete__(self, instance):
#         if instance in self.cache:
#             del self.cache[instance]
#
#     def __set__(self, instance, value):
#         self.cache[instance] = value
#
#     def __repr__(self):
#         return f'{self._instance}.data'
#
#     def __bool__(self):
#         return self._instance in self.cache
#
#     @property
#     def path(self) -> Path:
#         from validate_osm.compare.compare import Compare
#         self._instance: Compare
#         return (
#                 self._instance.directory /
#                 self.__class__.__name__ /
#                 f'{str(self._instance.bbox)}.feather'
#         )
#
#     def delete(self):
#         os.remove(self.path)
#

# footprints/
class DescriptorData:
    cache: WeakKeyDictionary[object, GeoDataFrame] = WeakKeyDictionary()

    def __init__(self):
        self._instance = None
        self._owner = None

    def __get__(self, instance: object, owner: Type) -> Union[GeoDataFrame, 'DescriptorData']:
        from validate_osm.compare.compare import Compare
        instance: Optional[Compare]
        owner: Optional[Type[Compare]]
        self._instance = instance
        self._owner = owner
        if instance is None:
            return self
        if instance in self.cache:
            return self.cache[instance]
        path = self.path
        # TODO: Perhaps if there is a file with a bbox that contains instance.bbox, load
        #   that bbox, .within it,
        if path.exists() and not 'data' not in instance.redo:
            with logged_subprocess(
                    instance.logger,
                    f'reading {owner.__name__}.data from {path} ({os.path.getsize(path) / 1024 / 1024 :.1f} MB)'
            ):
                data = self.cache[instance] = gpd.read_feather(path)
        else:
            with logged_subprocess(instance.logger, f'building {owner.__name__}.data'), self as data:
                with logged_subprocess(instance.logger, 'getting footprints'):
                    _ = self._instance.footprints
                with logged_subprocess(instance.logger, 'applying footprints'):
                    data = self.cache[instance] = self._instance.footprint(data)
        return data

    def __enter__(self) -> GeoDataFrame:
        # __enter__ is before the data is footprinted
        from validate_osm.compare.compare import Compare
        self._instance: Compare
        self._owner: Type[Compare]

        # We must concatenate DataFrames instead of going purely by Series because some columns
        #   may return a single-value, which must be repeated to the same length of other columns
        # To preserve SourceOSM.relation and SourceOSM.way indices, we must first
        def datas() -> Generator[gpd.GeoDataFrame, None, None]:
            for name, source in self._instance.sources.items():
                data: GeoDataFrame = source.data
                data = data.reset_index()
                data['name'] = name
                if 'id' not in data:
                    data['id'] = np.nan
                if 'group' not in data:
                    data['group'] = np.nan
                data = data.set_index(['name', 'id', 'group'], drop=True)
                yield data
                self._instance.logger.debug(f'deleting {source.resource.__class__.__name__}')
                del source.data

        with logged_subprocess(self._instance.logger, f'concatenating {self.__class__.__name__}.data', ):
            data = concat(datas())

        # TODO: Investigate how these invalid geometries came to be
        inval = data[data['geometry'].isna() | data['centroid'].isna()].index
        if len(inval):
            self._instance.logger.warning(f'no geom: {inval}')
            data = data.drop(index=inval)

        data['iloc'] = range(len(data))
        data['geometry'] = data['geometry'].to_crs(3857)
        data['centroid'] = data['centroid'].to_crs(3857)
        self.cache[self._instance] = data
        return data

    def __exit__(self, exc_type, exc_val, exc_tb):
        # __exit__ is after the data is footprinted
        from validate_osm.compare.compare import Compare
        self._instance: Compare
        data = self.cache[self._instance]
        path = self.path
        if not path.parent.exists():
            os.makedirs(path.parent)
        with logged_subprocess(self._instance.logger, f'serializing {self._owner.__name__}.data {path}'):
            data.to_feather(path)

    def __delete__(self, instance):
        if instance in self.cache:
            del self.cache[instance]

    def __set__(self, instance, value):
        self.cache[instance] = value

    def __repr__(self):
        return f'{self._instance}.data'

    @property
    def path(self) -> Path:
        from validate_osm.compare.compare import Compare
        self._instance: Compare
        return (
                self._instance.directory /
                self.__class__.__name__ /
                f'{str(self._instance.bbox)}.feather'
        )

    def delete(self):
        os.remove(self.path)
