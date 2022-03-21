import abc
import inspect
import os
from pathlib import Path
from typing import DefaultDict, Type, Union
from weakref import WeakKeyDictionary

import geopandas as gpd

from validate_osm.source import File
from validate_osm.util.scripts import logged_subprocess


class DescriptorPipe:
    _cache: DefaultDict[object, object]

    def __init__(self):
        self._instance = None
        self._owner = None

    def __get__(self, instance, owner):
        if instance in self._cache:
            return self._cache[instance]
        from validate_osm.source import Source
        self._instance: Source = instance
        self._owner: Type[Source] = owner
        if instance is None:
            return self
        return self._cache[instance]

    def __delete__(self, instance):
        del self._cache[instance]

    def __bool__(self):
        return self._instance in self._cache

    def __set__(self, instance, value):
        self._cache[instance] = value


class DescriptorPipeSerialize(DescriptorPipe, abc.ABC):
    _cache = WeakKeyDictionary[object, gpd.GeoDataFrame]
    name: str

    def __get__(self, instance: object, owner: Type):
        from validate_osm.source import Source
        instance: Union[Source, object]
        owner: Type[Source]
        self._instance = instance
        self._owner = owner
        if instance is None:
            return self
        if instance in self._cache:
            return self._cache[instance]
        path = self.path
        if not instance.ignore_file and path.exists():
            with logged_subprocess(
                    instance.logger,
                    f'reading {owner.__name__}.{self.name} from {path} ({File.size(path)})',
            ):
                data = self._cache[instance] = gpd.read_feather(path)
                return data

        with logged_subprocess(instance.logger, f'building {owner.__name__}.{self.name}'):
            data: gpd.GeoDataFrame = self._cache[instance]

        if not path.parent.exists():
            os.makedirs(path.parent)
        with logged_subprocess(instance.logger, f'serializing {owner.__name__}.{self.name} to {path}'):
            data.to_feather(path)
        return data

    @property
    def path(self) -> Path:
        return (
                Path(inspect.getfile(self._owner)).parent /
                'file' /
                self._owner.__name__ /
                self.__class__.__name__ /
                f'{str(self._instance.bbox)}.feather'
        )

    def delete_file(self) -> None:
        os.remove(self.path)
