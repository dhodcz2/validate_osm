import warnings
from shapely.geometry.base import BaseGeometry

import requests

warnings.filterwarnings('ignore', '.*PyGEOS.*')
from pathlib import Path
import abc
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
import networkx
import osmium
import pandas as pd

import functools
import itertools
import os
import math
import tempfile
from typing import Iterator

import pyrosm
import osmium
import concurrent.futures
import concurrent.futures
import multiprocessing
import warnings
from abc import ABC, abstractmethod
from typing import Type, Union, Optional, Iterable, Collection
from weakref import WeakKeyDictionary

import geopandas as gpd
import numpy as np
import osmium
import shapely.geometry.base
import osmium
from geopandas import GeoDataFrame
from geopandas import GeoSeries
from pandas import Series, DataFrame

from ..resource_ import DescriptorResource
from .handler import BaseHandler
from .suggest import Suggest
from ..static.file import StructFile, StructFiles, ListFiles
from ..bbox import BBox
from .preprocess import OSMPreprocessor

if False:
    from ..source import Source


class ResourceOsmium(DescriptorResource, ABC):
    Handler: Type[BaseHandler]
    suggest = Suggest()
    preprocessor = OSMPreprocessor()

    def __pbf(self, item: BBox) -> Iterable[str]:
        if isinstance(item, BBox):
            item = item.latlon
        pbf = self.suggest.cities(item, url=True)
        if pbf.size > 0:
            return pbf
        pbf = self.suggest.subregions(item, url=True)
        if pbf.size > 0:
            return pbf
        pbf = self.suggest.regions(item, url=True)
        if pbf.size > 0:
            return pbf
        pbf = self.suggest.countries(item, url=True)
        if pbf.size > 0:
            return pbf

    def __getitem__(self, item: BBox | BaseGeometry) -> ListFiles:
        pbf = self.__pbf(item)
        names = (
            url.rpartition('/')[2]
            for url in pbf
        )
        return ListFiles([
            StructFile(url=url, name=name, source=self._source)
            for url, name in zip(pbf, names)
        ])

    def __get__(self, instance: 'Source' | None, owner) -> BaseHandler | 'ResourceOsmium':
        if instance is None:
            return self
        self.source = instance
        self.Source = owner
        if not hasattr(instance, '_resource'):
            setattr(instance, '_resource', self.__resource())
        return instance._resource

    def __set__(self, instance, value: BaseHandler):
        instance._resource = value

    def __delete__(self, instance):
        if hasattr(instance, '_resource'):
            del instance._resource

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

    @classmethod
    @property
    def directory(self):
        return Path(__file__).parent / 'static'
