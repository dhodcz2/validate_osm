import abc
import functools
import itertools
from functools import lru_cache
from typing import Type, Any, Iterable, Iterator

import numpy as np
from geopandas import GeoSeries
from shapely.geometry.base import BaseGeometry

from .preprocess import StaticPreprocessor
from ..bbox import BBox
from ..resource_ import DescriptorResource
from .file import StructFile, ListFiles, StructFiles

if False:
    from ..source import Source


class DescriptorStaticResource(DescriptorResource, abc.ABC):
    preprocessor = StaticPreprocessor()

    @abc.abstractmethod
    def __getitem__(self, item: BBox | BaseGeometry) -> ListFiles:
        ...

    def __get__(self, instance: 'Source', owner: Type['Source']):
        if instance is None:
            return self
        self.source = instance
        self.Source = owner
        if hasattr(instance, '_resource'):
            if len(instance._resource) == 0:
                raise ValueError(f'len(instance._resource) == 0')
            self.__set__(instance, )
        return instance._resource

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


class StaticRegion(abc.ABC):
    @abc.abstractmethod
    @property
    def geometry(self) -> GeoSeries:
        """GeoSeries, where polygons represent the region boundary and the index is the region name."""

    @abc.abstractmethod
    def urls(self, names: Iterable[str]) -> Iterator[str | Iterable[str]]:
        """Maps region names to URLs."""

    def __set_name__(self, owner: 'DescriptorStaticRegions', name):
        self.name = name
        if 'regions' not in owner.__dict__:
            owner.regions = []
        owner.regions.append(self)

    def __get__(self, instance: 'DescriptorStaticRegions', owner):
        self.source = instance.source
        self.resource = instance

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
                    yield StructFile(name=name, url=url, source=self.source)
                elif isinstance(url, Iterable):
                    files = [
                        StructFile(name=name, url=u, source=self.source)
                        for u in url
                    ]
                    yield StructFiles(files=files, source=self.source, name=name)
                else:
                    raise TypeError(url)

        return ListFiles(list(structs()))


class DescriptorStaticRegions(DescriptorStaticResource, abc.ABC):
    regions: list['StaticRegion']

    @lru_cache(1)
    def __getitem__(self, item: BBox) -> ListFiles:
        files = list(itertools.chain.from_iterable(region[item] for region in self.regions))
        return ListFiles(files)

    @lru_cache(1)
    def __contains__(self, item):
        return any(item in region for region in self.regions)
