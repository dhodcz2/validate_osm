import abc
import concurrent.futures
import dataclasses
import functools
import inspect
import logging
import os
from functools import lru_cache
from pathlib import Path
from typing import Any, Type
from typing import Union, Iterable, Iterator
from weakref import WeakKeyDictionary

import geopandas as gpd
import requests
from geopandas import GeoDataFrame
from geopandas import GeoSeries
from shapely.geometry.base import BaseGeometry

from validate_osm.logger import logger, logged_subprocess
from validate_osm.util.scripts import concat
from validate_osm.source.bbox import BBox
# from validate_osm.source.preprocessor import DescriptorPreprocessor
from validate_osm.source.preprocessor import CallablePreprocessor

if False:
    from validate_osm.source.source import Source


class Resource(abc.ABC):
    name: str
    boundary: BBox
    crs: Any
    flipped: bool = False
    preprocess: CallablePreprocessor

    @abc.abstractmethod
    def __contains__(self, item: Union[BBox]) -> bool:
        ...

    def __init__(self, *args, **kwargs):
        # TODO: Is there a more efficient way of doing this per class instead of per instance?
        #   probably not without metaclass...
        self.bbox_to_geom = lru_cache(1)(self.bbox_to_geom)
        self.source = None
        self.owner = None

    def bbox_to_geom(self, bbox: BBox) -> BaseGeometry:
        return bbox.to_crs(self.crs).ellipsoidal if self.flipped else bbox.to_crs(self.crs).cartesian

    @classmethod
    @property
    def directory(cls) -> Path:
        return Path(inspect.getfile(cls)).parent / 'static' / 'resource' / cls.__name__


@dataclasses.dataclass(repr=False)
class StructFile:
    url: str = dataclasses.field(repr=False)
    source: 'Source' = dataclasses.field(repr=False)
    name: Union[str, Iterable[str], None] = dataclasses.field(repr=False, default=None)

    def __repr__(self):
        return f'{self.source.name}.{self.name}'

    def __post_init__(self):
        if self.name is None:
            self.name = self.url.rpartition('/')[2].rpartition('.')[0]
        self._resource = None


    @property
    def resource(self) -> Path:
        if self._resource is not None:
            return self._resource
        return self.source.__class__.resource.directory / self.url.rpartition('/')[2]

    @resource.setter
    def resource(self, value):
        self._resource = value

    @functools.cached_property
    def data(self) -> Path:
        self.source: 'Source'
        name = self.name
        path = self.source.__class__.data.directory
        if name is None:
            raise AttributeError(self.name)
        elif isinstance(name, str):
            return path / f'{name}.feather'
        elif isinstance(name, (tuple, list)):
            for sub in name[:-1]:
                path /= sub
            return path / f'{name[-1]}.feather'
        else:
            raise TypeError(name)

    def load_resource(self) -> GeoDataFrame:
        # TODO: Can we limit columns to only those that are relevant?
        path = self.resource
        with logged_subprocess(f'reading {path}', level=logging.DEBUG):
            match path.name.rpartition('.')[2]:
                case 'feather':
                    gdf = gpd.read_feather(path)
                case 'parquet':
                    gdf = gpd.read_parquet(path)
                case _:
                    gdf = gpd.read_file(path)
            return gdf

    # def transform(self) -> None:
    #     self.instance.resource = self.load_resource()
    #     self.instance.data.to_feather(self.source)

    def load_source(self, columns=None, **kwargs) -> GeoDataFrame:
        return gpd.read_feather(self.data, columns, **kwargs)

    def rm_source(self):
        os.remove(self.data)

    def rm_resource(self):
        os.remove(self.resource)


@dataclasses.dataclass(repr=False)
class StructFiles:
    files: list[StructFile] = dataclasses.field(repr=False)
    source: 'Source' = dataclasses.field(repr=False)
    name: Union[str, Iterable[str]] = dataclasses.field(repr=False)

    @functools.cached_property
    def data(self) -> Path:
        name = self.name
        path = self.source.__class__.data.directory
        if name is None:
            raise AttributeError(self.name)
        elif isinstance(name, str):
            return path / f'{name}.feather'
        elif isinstance(name, (tuple, list)):
            for sub in name[:-1]:
                path /= sub
            return path / f'{name[-1]}.feather'
        else:
            raise TypeError(name)

    def load_resource(self) -> GeoDataFrame:
        gdfs: Iterator[GeoDataFrame] = (
            file.load_resource()
            for file in self.files
        )
        return concat(gdfs)

    def load_source(self, colunns=None, **kwargs) -> GeoDataFrame:
        return gpd.read_feather(self.data, colunns, **kwargs)

    def __repr__(self):
        return f'{self.source.name}.{self.name}'


class DescriptorStatic(Resource):
    from validate_osm.source.preprocessor import CallableStaticPreprocessor
    preprocess = CallableStaticPreprocessor()
    cache: WeakKeyDictionary[object, GeoDataFrame] = WeakKeyDictionary()

    def __init__(self, *args, **kwargs):
        super(DescriptorStatic, self).__init__(*args, **kwargs)
        self.__getitem__ = lru_cache()(self.__getitem__)

    @abc.abstractmethod
    def __getitem__(self, item: Union[BaseGeometry, BBox]) -> list[Union[StructFile, StructFiles]]:
        ...

    # def __iter__(self) -> Iterator[GeoDataFrame]:
    #     bbox = self.source.bbox
    #     files = self[bbox]
    #     for file in files:
    #         yield file.load_resource()

    def __set__(self, instance, value: GeoDataFrame):
        if instance is None:
            raise ValueError(instance)
        del instance.data
        self.cache[instance] = value

    def __get__(self, instance: 'Source', owner: Type['Source']) -> Union['DescriptorStatic', GeoDataFrame]:
        if instance in self.cache:
            return self.cache[instance]
        self.source = instance
        self.owner = owner

        # Case: Source.__class__.resource
        if instance is None:
            return self

        # Case: Compare.source.resource
        if instance.compare is not None:
            return self

        # Case: Source.resource
        if instance.compare is None:
            self.__set__(instance, concat(self))

        return self.__get__(instance, owner)

    def __iter__(self) -> Iterator[GeoDataFrame]:
        bbox = self.source.bbox.to_crs(3857).ellipsoidal
        for file in self[self.source.bbox]:
            resource = file.load_resource()
            resource = resource[resource.intersects(bbox)]
            yield resource

    def __delete__(self, instance):
        del self.cache[instance]

    @classmethod
    @property
    def directory(cls) -> Path:
        return Path(inspect.getfile(cls)).parent / 'static' / 'resource' / cls.__name__


class DescriptorStaticNaive(DescriptorStatic):
    def __init__(
            self,
            file: Union[StructFile, StructFiles],
            crs: Any,
            name: str,
            link: str,
            flipped=False,
            unzipped=None,
            *args,
            **kwargs
    ):
        super(DescriptorStaticNaive, self).__init__(*args, **kwargs)
        self._files: list[Union[StructFile, StructFiles]] = [file]
        self.crs = crs
        self.name = name
        self.link = link
        self.flipped = flipped
        self.unzipped = unzipped

    @property
    def files(self) -> list[StructFile, StructFiles]:
        return [
            file.__class__(**file.__dict__, instance=self.source)
            for file in self._files
        ]

    @lru_cache(1)
    def __contains__(self, bbox: BBox) -> bool:
        if not issubclass(bbox.__class__, BBox):
            raise TypeError(bbox)
        bbox = self.bbox_to_geom(bbox)
        gdfs = (
            file.load_source()
            for file in self.files
        )
        return any(
            any(gdf.intersects(bbox))
            for gdf in gdfs
        )

    @lru_cache(1)
    def __getitem__(self, item: BBox, ) -> list[Union[StructFile, StructFiles]]:
        # TODO: is it worth it to load the entire dataset and check if the BBox contains it?
        return self.files


class DescriptorStaticRegions(DescriptorStatic, abc.ABC):
    class StaticRegion(abc.ABC):
        menu: set[str]
        boundary: BaseGeometry

        @abc.abstractmethod
        @functools.cached_property
        def geometry(self) -> GeoSeries:
            """

            :return: GeoSeries with columns as the names that will feed into self.files that creates files from names

            """

        @abc.abstractmethod
        def urls(self, names: Iterable[str]) -> Iterator[str]:
            ...

        # TODO: How can this be structured such that StructFiles is overwritten?
        #   inherit from StaticRegino and overwrite getitem
        @lru_cache(1)
        def __getitem__(self, bbox: BBox) -> Iterator[Union[StructFile, StructFiles]]:
            if not issubclass(bbox.__class__, BBox):
                raise TypeError(bbox)
            bbox = bbox.to_crs(self.geometry.crs).cartesian
            names = self.geometry[self.geometry.intersects(bbox)].index
            urls = self.urls(names)

            for name, url in zip(names, urls):
                if isinstance(url, str):
                    yield StructFile(url, self.source, name)
                elif isinstance(url, Iterable):
                    yield StructFiles(
                        files=[
                            StructFile(item, self.source, name)
                            for item in url
                        ],
                        source=self.source,
                        name=name
                    )
                else:
                    raise TypeError(url)

        @lru_cache(1)
        def __contains__(self, bbox: BBox):
            if not issubclass(bbox.__class__, BBox):
                raise TypeError(bbox)
            bbox = bbox.to_crs(self.geometry.crs).cartesian
            return any(self.geometry.intersects(bbox))

        def __init__(self, instance: 'DescriptorStaticRegions'):
            self.resource = instance

        @property
        def source(self):
            return self.resource.source

    def __init__(self, *args, **kwargs):
        if not issubclass(self.__class__, DescriptorStaticRegions):
            raise TypeError(self)
        super(DescriptorStaticRegions, self).__init__(*args, **kwargs)
        self.regions = [
            attribute(self)
            for attribute in self.__class__.__dict__.values()
            if isinstance(attribute, type) and issubclass(attribute, DescriptorStaticRegions.StaticRegion)
        ]

    def __delete__(self, instance):
        if instance in self.cache:
            del self.cache[instance]
        for region in self.regions:
            if 'geometry' in region.__dict__:
                del region.geometry

    @lru_cache(1)
    def __contains__(self, bbox: BBox):
        return any(
            bbox in region
            for region in self.regions
        )

    @lru_cache(1)
    def __getitem__(self, item: BBox) -> list[Union[StructFile, StructFiles]]:
        return [
            file
            for region in self.regions
            if item in region
            for file in region[item]
        ]
