from weakref import WeakKeyDictionary
import geopandas as gpd
import abc
import inspect
import os
from pathlib import Path
from typing import DefaultDict, Type, Optional


class DescriptorPipe:
    _cache: DefaultDict[object, object]

    def __init__(self):
        self._instance = None
        # self._owner = None

    def __get__(self, instance, owner):
        if instance in self._cache:
            return self._cache[instance]
        from ValidateOSM.source import Source
        self._instance: Source = instance
        self._owner: Type[Source] = owner
        # self._instance = instance
        # self._owner = owner
        if instance is None:
            return self
        return self._cache[instance]

    def __delete__(self, instance):
        del self._cache[instance]

    def __bool__(self):
        return self._instance in self._cache


class DescriptorPipeSerialize(DescriptorPipe, abc.ABC):
    _cache = WeakKeyDictionary[object, gpd.GeoDataFrame]

    def __get__(self, instance, owner):
        from ValidateOSM.source import Source
        self._instance: Source = instance
        self._owner: Type[Source] = owner
        if instance is not None and instance not in self._cache:
            path = self.path
            if path.exists():
                self._cache[instance] = gpd.read_feather(path)
            else:
                if not path.parent.exists():
                    os.makedirs(path.parent)
                self._cache[instance].to_feather(path)
            return self._cache[instance]
        return super(DescriptorPipeSerialize, self).__get__(instance, owner)

    @property
    def path(self) -> Path:
        return (
            Path(inspect.getfile(self._owner)).parent /
            'resources' /
            self._owner.__name__ /
            f'{self.__class__.__name__}.feather'
        )

    def delete_file(self) -> None:
        os.remove(self.path)
