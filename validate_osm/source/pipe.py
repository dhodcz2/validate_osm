import logging
from weakref import WeakKeyDictionary
import geopandas as gpd
import abc
import inspect
import os
from pathlib import Path
from typing import DefaultDict, Type, Optional, Union


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
        # TODO: Never use self._instance from __get__; in the debugger it causes weirdness
        # from validate_osm.source import Source
        # self._instance: Source = instance
        # self._owner: Type[Source] = owner
        # if instance is not None and instance not in self._cache:
        #     path = self.path
        #     if path.exists() and not instance.ignore_file:
        #         self._cache[instance] = gpd.read_feather(path)
        #     else:
        #         if not path.parent.exists():
        #             os.makedirs(path.parent)
        #         self._cache[instance].to_feather(path)
        #     return self._cache[instance]
        # return super(DescriptorPipeSerialize, self).__get__(instance, owner)

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
            logging.info(f'reading {owner.__name__}.{self.name} from {path}')
            data = self._cache[instance] = gpd.read_feather(path)
            return data
        logging.info(f'building {owner.__name__}.{self.name}')
        data: gpd.GeoDataFrame = self._cache[instance]
        if not path.parent.exists():
            os.makedirs(path.parent)
        logging.info(f'serializing {owner.__name__}.{self.name} to {path}')
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
