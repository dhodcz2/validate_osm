import abc
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

from .bbox import BBox


class CallablePreprocessor(abc.ABC):
    """
    Preprocessors are Singletons that take a set of Sources, and provide ETL ops for Source.data,
    leveraging multithreading and or multiprocessing.
    """

    @abc.abstractmethod
    def __call__(self, *args: 'Source', **kwargs):
        ...

    def __new__(cls, *args, **kwargs):
        if not hasattr(cls, '_instance'):
            cls._instance = super().__new__(cls)
        return cls._instance


class DescriptorResource(abc.ABC):
    name: str
    boundary: BBox
    preprocessor: CallablePreprocessor

    @abc.abstractmethod
    def __get__(self, instance, owner):
        ...

    @abc.abstractmethod
    def __contains__(self, item: BBox) -> bool:
        ...

    @abc.abstractmethod
    def __bool__(self):
        ...

    @abc.abstractmethod
    def __set__(self, instance, value):
        ...

    @classmethod
    @property
    def data(cls) -> Path:
        return Path(inspect.getfile(cls)).parent / 'static' / 'data' / cls.name

    @classmethod
    @property
    def raw(cls) -> Path:
        return Path(inspect.getfile(cls)).parent / 'static' / 'raw' / cls.name
