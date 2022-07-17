from geopandas import GeoDataFrame, GeoSeries
import numpy as np

from pandas import Series, DataFrame

import inspect
import functools
from pathlib import Path
from functools import lru_cache
import itertools
from typing import Type, Any, Iterable, Iterator

from shapely.geometry.base import BaseGeometry
import abc

from ..bbox import BBox
from ..resource_ import DescriptorResource
from .preprocess import StaticPreprocessor
from .file import StructFile, StructFiles, ListFiles

if False:
    from ..source import Source


class DescriptorStaticResource(DescriptorResource, abc.ABC):
    preprocessor = StaticPreprocessor()

    @abc.abstractmethod
    def __getitem__(self, item: BBox | BaseGeometry) -> ListFiles:
        ...

    def __get__(self, instance: 'Source', owner: Type['Source']):
        if hasattr(instance, '_resource'):
            if len(instance._resource) == 0:
                raise ValueError(f'len(instance._resource) == 0')
            return instance._resource
        self.source = instance
        self.Source = owner
        return self

    def __bool__(self):
        files = self[self.source.bbox]
        if all(file.exists() for file in files.data_):
            self.source.data = files.load(files.data_)
            return False  # no need to load self.resource;
        else:
            self.source.preprocess()

        if not all(file.exists() for file in files.data_):
            raise RuntimeError(f'Somehow, the data files do not exist after preprocessing')
        return False

    def __set__(self, instance, value):
        if instance is None:
            raise TypeError(instance)
        setattr(instance, '_resource', value)

    def __delete__(self, instance):
        if hasattr(instance, '_resource'):
            del instance._resource


class DescriptorStaticNaive(DescriptorStaticResource, abc.ABC):
    def __init__(
            self,
            files: ListFiles,
            crs: Any,
            name: str,
            link: str,
            boundary: BBox,
    ):
        self.files = files
        self.crs = crs
        self.name = name
        self.link = link
        self.boundary = boundary

    @lru_cache(1)
    def __getitem__(self, item: BBox | BaseGeometry) -> ListFiles:
        return self.files

    @lru_cache(1)
    def __contains__(self, item: BBox):
        return self.boundary.latlon.intersects(item.to_crs(self.boundary.crs))


class DescriptorStaticRegions(DescriptorStaticResource, abc.ABC):

    @lru_cache(1)
    def __getitem__(self, item: BBox) -> ListFiles:
        ...


class RegionMeta(abc.ABCMeta):
    def __init__(self, name, bases, local):
        super(RegionMeta, self).__init__(name, bases, local)
        local['geometry'] = functools.cached_property(local['geometry'])


class StaticRegion(abc.ABC, metaclass=RegionMeta):
    @abc.abstractmethod
    @property
    def geometry(self) -> GeoSeries:
        ...

    @abc.abstractmethod
    def urls(self, names: Iterable[str]) -> Iterator[str]:
        ...

    @lru_cache(1)
    def __contains__(self, item: BBox):
        intersects: np.ndarray = self.geometry.intersects(item.to_crs(self.geometry.crs).lonlat)
        return intersects.any()

    @lru_cache(1)
    def __getitem__(self, item: BBox) -> ListFiles:
        geometry = self.geometry
        names = geometry[geometry.intersects(item.to_crs(geometry.crs).lonlat)].index
        urls = self.urls(names)

        def structs() -> Iterator[StructFile | StructFiles]:
            for name, url in zip(names, urls):
                if isinstance(url, str):
                    yield StructFile(name, url)
                elif isinstance(url, Iterable):
                    yield StructFiles(name, url)
                else:
                    raise TypeError(url)


