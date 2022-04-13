import abc
from validate_osm.source.resource_ import Resource
import functools
import inspect
import itertools
import warnings
from pathlib import Path
from typing import Type, Iterator, Optional
from typing import Union

import numpy as np
import pandas as pd
from geopandas import GeoDataFrame

from validate_osm.source.aggregate import FactoryAggregate
from validate_osm.source.bbox import BBox
from validate_osm.source.data import DecoratorData, DescriptorData
from validate_osm.source.footprint import CallableFootprint

from validate_osm.source.resource_ import (
    DescriptorStatic,
    DescriptorStaticRegions,
    DescriptorStaticNaive,
    StructFile,
    StructFiles
)

if False | False:
    from validate_osm import Compare

warnings.filterwarnings('ignore', message='.*initial implementation of Parquet.*')


class SourceMeta(abc.ABCMeta, type):
    def __new__(cls, name, bases, local):
        object = super(SourceMeta, cls).__new__(cls, name, bases, local)
        if '_data' not in local:
            setattr(object, '_data', {})
        return object


class Source(abc.ABC, metaclass=SourceMeta):
    resource: Union[GeoDataFrame, DescriptorStaticRegions, DescriptorStaticNaive]
    data: Union[DescriptorData, GeoDataFrame] = DescriptorData()
    footprint: Type[CallableFootprint] = CallableFootprint
    aggregate_factory: Type[FactoryAggregate] = FactoryAggregate
    name: str
    link: str

    def preprocess(self):
        self.resource.preprocess(self)

    def __init__(self, redo=False, bbox=None, serialize=False, compare: Optional['Compare'] = None):
        # TODO
        # abstracts = [
        #     struct
        #     # for struct in self.__class__.data.structs.load.values()
        #     if struct.abstract
        # ]
        # if abstracts:
        #     raise TypeError(f"{self.__class__.__name__} inherited abstract methods for its data: {abstracts}")
        self.redo = redo
        self.bbox = bbox
        self.serialize = serialize
        self.compare = compare

    def __contains__(self, item: BBox) -> bool:
        return item in self.__class__.resource

    def __iter__(self) -> Iterator[GeoDataFrame]:
        bbox = self.bbox.to_crs(3857).ellipsoidal
        for file in self.resource[self.bbox]:
            data = file.load_source()
            data = data[data.intersects(bbox)]
            yield data

    @property
    def redo(self):
        if self.compare is not None:
            return self.name in self.compare.redo
        else:
            return self._redo

    @redo.setter
    def redo(self, val: bool):
        self._redo = val

    @classmethod
    @property
    def name(cls) -> str:
        return cls.resource.name

    @classmethod
    @property
    def link(cls) -> str:
        return cls.resource.link

    def group(self) -> GeoDataFrame:
        """
        Assign to self.data an index
        :return: None
        """
        data = self.data
        return data.set_index(pd.Index(data=itertools.repeat(np.nan, len(data)), name='group'), append=True)

    def resource(self) -> Union[DescriptorStatic]:
        """An instance or Iterator of instances that encapsulate the raw data that is entering this pipeline."""

    @classmethod
    @property
    def validating(cls) -> set[str]:
        """The specific columns that will be validated"""
        return cls.data.validating

    @DecoratorData(dtype='geometry', crs=3857)
    @abc.abstractmethod
    def geometry(self):
        """The geometry that the data entry represents"""

    @DecoratorData(dtype='datetime64[ns]')
    @abc.abstractmethod
    def timestamp(self):
        """The time of the data entry"""

    @DecoratorData(dtype='geometry', crs=3857, dependent={'geometry'})
    def centroid(self):
        loc = self.data['geometry'].notna()
        return self.data.loc[loc, 'geometry'].centroid

    @DecoratorData(dtype='string', crs=None, dependent='centroid')
    def ref(self):
        return pd.Series((
            f'{centroid.y:.4f}, {centroid.x:.4f}'
            if centroid is not None
            else None
            for centroid in self.data['centroid'].to_crs(4326)
        ), index=self.data.index, dtype='string')

        # loc = self.data['centroid'].notna()
        # return GeoSeries((
        #     shapely.geometry.Point(centroid.y, centroid.x)
        #     for centroid in self.data.loc[loc, 'centroid'].to_crs(4326)
        # ), index=self.data.loc[loc].index)

    def __hash__(self):
        return hash(self.name)
