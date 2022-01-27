import os
from collections import UserDict
import geopandas as gpd
from weakref import WeakKeyDictionary
from pathlib import Path
from weakref import WeakKeyDictionary

import numpy as np
import pandas as pd
from annoy import AnnoyIndex
from itertools import chain

from pandas import Series

from typing import Iterator, Collection, Union, Iterable, Type, Optional, Generator

import geopandas as gpd

import dataclasses
from pathlib import Path

import functools
import inspect
import warnings

import pandas
from geopandas import GeoDataFrame, GeoSeries
from sys import _getframe
from pandas import Series
from typing import Callable, Iterator, Union, Any, Type


class DecoratorAggregate:
    @dataclasses.dataclass
    class Struct:
        name: str
        func: Callable
        crs: Any
        dtype: Any
        dependent: Union[set[str], str]
        _func: Callable = dataclasses.field(init=False, repr=False)

        def __post_init__(self):
            self.dependent = {self.dependent} if isinstance(self.dependent, str) else set(self.dependent)

        def __hash__(self):
            return hash(self.name)

        @property
        def func(self) -> Callable:
            return self.decorate

        @func.setter
        def func(self, func: Callable):
            self._func = func

        def decorate(self) -> Callable:
            @functools.wraps(self._func)
            def wrapper(source: type, groups: object):
                obj: object = self._func(source, groups)
                if self.dtype == 'geometry':
                    if self.crs is None:
                        obj = GeoSeries(obj, crs=None)
                    else:
                        if not isinstance(obj, GeoSeries):
                            raise TypeError(f"{obj} must be returned as a GeoSeries so that it may specify a CRS.")
                        if obj.crs is None:
                            raise ValueError(f"{obj} must specify a CRS.")
                        obj = obj.to_crs(self.crs)
                if isinstance(obj, Series):
                    if not isinstance(obj.index, pandas.core.indexes.range.RangeIndex):
                        warnings.warn(
                            f"{obj}.index returns a {type(obj.index)}; naively passing this may result in a mismatched "
                            f"column index. Resetting this index so that column indices align regardless of implementation."
                        )
                        obj = obj.reset_index(drop=True)
                    if self.dtype:
                        obj = obj.astype(self.dtype)
                else:
                    obj = Series(obj, dtype=self.dtype)

                return obj

            return wrapper

    def __init__(self, name: str, dtype=None, crs=None, dependent: Union[str, set[str]] = None):
        self.name = name
        self.dtype = dtype
        self.crs = crs
        self.dependent = {dependent} if isinstance(dependent, str) else set(dependent)

    def __call__(self, func: Callable):
        frame = inspect.currentframe()
        frames = inspect.getouterframes(frame)
        frame = frames[1]
        _aggregate: dict[str, DecoratorAggregate.Struct] = frame.frame.f_locals.setdefault('_aggregate', {})
        _aggregate[self.name] = DecoratorAggregate.Struct(
            name=self.name,
            dtype=self.dtype,
            crs=self.crs,
            dependent=self.dependent,
            func=func
        )
        return func


class CacheStructs:
    def __init__(self):
        from ValidateOSM.source.source import Source
        self._Source = Source
        super(CacheStructs, self).__init__()
        self.data: WeakKeyDictionary[object, DecoratorAggregate.Struct] = WeakKeyDictionary()

    def __missing__(self, source: type):
        Source = self._Source
        sources: Iterator[Type[Source]] = (s for s in source.mro()[::-1] if issubclass(s, Source))
        structs: set[DecoratorAggregate.Struct] = getattr(source, '_aggregate')
        structs.update((
            struct
            for source in sources
            for struct in self.data[source]
        ))
        self.data[source] = structs
        return structs


class CacheAggregate(UserDict):
    _cache_structs: WeakKeyDictionary[type, set[DecoratorAggregate.Struct]] = CacheStructs()

    def __init__(self):
        from ValidateOSM.source.source import Source
        self._Source = Source
        super(CacheAggregate, self).__init__()
        self.data: WeakKeyDictionary[object, GeoDataFrame] = WeakKeyDictionary()

    def __missing__(self, instance: object):
        structs = self._cache_structs[instance.__class__]
        indie = {
            struct.name: struct.func(instance)
            for struct in structs
            if not struct.dependent
        }
        rows = max(len(s) for s in indie.values())
        agg = GeoDataFrame({
            n: (s.repeat(rows) if len(s) == 1 else s)
            for n, s in indie.items()
        })
        self.data[instance.__class__] = agg
        depend: set[DecoratorAggregate.Struct] = {struct for struct in structs if struct.dependent}
        while depend:
            viable: set[DecoratorAggregate.Struct] = {
                struct for struct in depend if not struct.dependent.difference(agg.columns)
            }
            if not viable:
                raise RuntimeError(f"Cannot resolve with cross-dependencies: {depend}")
            for struct in viable:
                series = struct.func(instance)
                agg[struct.name] = series.repeat(rows) if len(series) == 1 else series
            depend = {struct for struct in depend if struct not in viable}
        agg.index.name = 'i'
        return agg


class DescriptorAggregate:
    _cache_aggregate: CacheAggregate[object, GeoDataFrame] = CacheAggregate()

    def __init__(self):
        self._instance = None
        self._owner = None

    def __get__(self, instance, owner) -> Union['DecoratorAggregate', GeoDataFrame]:
        if instance is not None:
            try:
                return self._cache_aggregate.setdefault(instance, self.from_file())
            except FileNotFoundError as e:
                raise FileNotFoundError(
                    f"Attempted to call {self._instance}.aggregate.from_file() before it has been serialized; "
                    f"please perform an (in)dependent aggregate with {self._owner}.aggregate.__call__"
                ) from e
        if self._owner is not owner:
            self._owner = owner
            self._instance = self._owner()
        return self

    def __call__(self, source: Optional[type] = None) -> None:
        from ValidateOSM.source.source import Source
        source: Source
        self._instance: Source
        if source is None:
            self._instance: Source
            self._cache_aggregate[self._instance].to_feather(self.path)
        else:
            Source = self._Source
            source: Type[Source]
            self._instance: Source
            instance: Source = source()
            self._instance.groups(instance)
            self._cache_aggregate[instance].to_feather(source.aggregate.path)
        return None

    def __delete__(self, instance: object):
        del self._cache_aggregate[instance]

    def from_file(self) -> GeoDataFrame:
        return gpd.read_feather(self.path)

    @property
    def path(self) -> Path:
        cls = self._owner
        path = Path(inspect.getfile(cls)).parent / 'aggregate'
        if not path.exists():
            path.mkdir()
        path = path / f"{cls.__name__}.feather"
        return path

    def delete_file(self):
        os.remove(self.path)
