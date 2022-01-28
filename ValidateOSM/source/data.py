import contextlib
from collections import UserDict
import os
from pathlib import Path

import pandas as pd
from geopandas import GeoDataFrame
import inspect

import pyproj.exceptions
from geopandas import GeoSeries
import warnings

import geopandas as gpd
from pandas.core.indexes.range import RangeIndex
import abc
import dataclasses
import functools
import itertools
from sys import _getframe
from typing import Callable, Any, Optional, Iterator, Iterable, Collection, Union, Type
from collections import ChainMap

import pandas
from pandas import Series, DataFrame
from weakref import WeakKeyDictionary


class _DecoratorData:
    @dataclasses.dataclass
    class Struct:
        name: str
        func: Callable
        dtype: str
        subtype: type
        abstract: bool
        crs: Any
        dependent: set[str]
        _func: Callable = dataclasses.field(init=False, repr=False)

        def __hash__(self):
            return hash(self.name)

        def __eq__(self, other):
            return self.name == other

        @property
        def func(self) -> Callable:
            return self.decorate

        @func.setter
        def func(self, func: Callable):
            self._func = func

        def decorate(self):
            @functools.wraps(self._func)
            def wrapper(source: object):
                obj: object = self._func(source)
                # Expect a GeoSeries
                if self.dtype == 'geometry':
                    if self.crs is None:
                        obj = GeoSeries(obj, crs=None)
                    else:
                        if not isinstance(obj, GeoSeries):
                            raise TypeError(f"{obj} must be returned as a GeoSeries so that it may specify a CRS.")
                        if obj.crs is None:
                            raise ValueError(f"{obj} must specify a CRS.")
                        obj = obj.to_crs(self.crs)

                # Got a Series
                if isinstance(obj, Series):
                    if not isinstance(obj.index, pandas.core.indexes.range.RangeIndex):
                        warnings.warn(
                            f"{obj}.index returns a {type(obj.index)}; naively passing this may result in a mismatched "
                            f"column index. Resetting this index so that column indices align regardless of implementation."
                        )
                        obj = obj.reset_index(drop=True)
                    if self.dtype is not None:
                        obj = obj.astype(self.dtype)
                else:
                    obj = Series(obj, dtype=self.dtype)
                return obj

            return wrapper

    def __init__(self, dtype, crs=None, dependent=Union[str, set[str]]):
        self.dtype = dtype
        self.crs = crs
        self.dependent = {dependent} if isinstance(dependent, str) else set(dependent)

    args = ['self']

    def __call__(self, func: Callable) -> Callable:
        argspec = inspect.getfullargspec(func)
        if argspec.args != self.args:
            raise SyntaxError(
                f"{func.__name__} has argspec of {argspec}; {self.__class__.__name__} expects {self.args}"
            )
        frame = inspect.currentframe()
        frames = inspect.getouterframes(frame)
        frame = frames[1]
        _data: dict[str, _DecoratorData.Struct] = frame.frame.f_locals.setdefault('_data', {})
        name = func.__name__
        _data[name] = _DecoratorData.Struct(
            name=func.__name__,
            func=func,
            dtype=self.dtype,
            subtype=self.__class__,
            abstract=getattr(func, '__isabstractmethod__', False),
            crs=self.crs,
            dependent=self.dependent
        )
        return func


class DecoratorData(_DecoratorData):
    """
    Wraps a function as @data to be extracted into a Series that is standardized and uniform across different
        datasets.
    """

    class identifier(_DecoratorData):
        """
        In addition to wrapping a function as @data, this marks a particular column as being able to be used as a
        identifier to indicate relationships across datasets.
        """

    class validate(_DecoratorData):
        """
        In addition to wrapping a function as @data, this marks a particular column as one that will be validated as an
        end goal;
        """


class CacheStructs(UserDict):
    """
    Inherits structures from the Source's MRO which will determine the attributes of the Series to be extracted
    in the process of self.raw to self.data
    """

    def __missing__(self, source: type) -> dict[str, DecoratorData.Struct]:
        from ValidateOSM.source import Source
        sources: Iterator[Type[Source]] = [
            s for s in source.mro()[::-1]
            if issubclass(s, Source)
        ]
        implications: list[dict[str, DecoratorData.Struct]] = [
            self.data[source]
            if source in self.data
            else getattr(source, '_data')
            for source in sources
        ]
        implications.reverse()  # Linear if constructed right-to-left; x^2 if constructed left-to-right
        names: set[str] = {
            name
            for implication in implications
            for name in implication.keys()
        }

        def inherit(name: str) -> DecoratorData.Struct:
            structs: Iterator[DecoratorData.Struct] = (
                imp[name]
                for imp in implications
                if name in imp
            )
            struct = next(structs)
            dtype = struct.dtype
            subtype = struct.subtype
            crs = struct.crs
            dependent = struct.dependent
            if not struct.abstract:
                func = struct._func
                abstract = True
            else:
                try:
                    func = next(
                        struct.func
                        for struct in structs
                        if not struct.abstract
                    )
                except StopIteration as e:
                    if isinstance(source, abc.ABC):
                        func = None
                        abstract = True
                    else:
                        raise NotImplementedError(
                            f"{source.__name__}.{name} inheritance could not be resolved."
                        ) from e
                else:
                    abstract = False
            return DecoratorData.Struct(
                name=name,
                func=func,
                dtype=dtype,
                subtype=subtype,
                abstract=abstract,
                crs=crs,
                dependent=dependent
            )

        result = {name: inherit(name) for name in names}
        self.data[source] = result
        return result


class CacheData(UserDict):
    """
        Handles the extraction of standardized, interchangeable data from the raw data source set to the attribute of
        self.raw within Source.__init__
    """
    _cache_structs: CacheStructs[type, dict[str, DecoratorData.Struct]] = CacheStructs()

    def __init__(self):
        super(CacheData, self).__init__()
        self.data: WeakKeyDictionary[object, GeoDataFrame] = WeakKeyDictionary()

    def __missing__(self, key):
        self._structs = self._cache_structs[key]
        self._instance = key
        self._source = key.__class__
        data = self.resolve()
        self.data[key] = data
        return data

    @contextlib.contextmanager
    def subsource(self, subsource: object) -> None:
        sources = self._instance.raw
        self._instance.raw = subsource
        yield
        self._instance.raw = sources

    def resolve(self) -> GeoDataFrame:
        from ValidateOSM.source.source import Source
        instance: Source = self._instance
        self._source: Type[Source]
        if isinstance(self._instance.raw, Iterator):
            sources: Iterator[GeoDataFrame] = (self.resolve_subsource(sub) for sub in instance.raw)
            data = instance.concat(sources)
        else:
            data = self.resolve_subsource(instance.raw)
        data: GeoDataFrame = data.reset_index()
        loc = ~data['geometry'].is_valid
        data.loc[loc, 'geometry'] = data.loc[loc, 'geometry'].buffer(0)
        # data = data.set_geometry('geometry')
        return data

    def resolve_subsource(self, subsource: object) -> GeoDataFrame:
        with self.subsource(subsource):
            # Extract columns that do not have data dependencies
            indie = {
                name: struct.func(self._instance)
                for name, struct in self._structs.items()
                # for name, struct in self._instance.it
                # for name, struct in self._structs.items()
                if not struct.dependent
            }
            rows = max(len(obj) for obj in indie.values())
            data = GeoDataFrame({
                name: (series.repeat(rows) if len(series) == 1 else series)
                for name, series in indie.items()
            })

            # Extract data that is dependent on data that has been extracted.
            self.data[self._instance] = data
            depend: set[DecoratorData.Struct] = {
                struct
                for struct in self._structs.values()
                if struct.dependent
            }
            while depend:
                viable: set[DecoratorData.Struct] = {
                    struct for struct in depend
                    if not struct.dependent.difference(data.columns)  # If there is no dif, then all the req are avail
                }
                if not viable:
                    raise RuntimeError(f"Cannot resolve with cross-dependencies: {depend}")
                for struct in viable:
                    series = struct.func(self._instance)
                    data[struct.name] = series.repeat(rows) if len(series) == 1 else series
                depend.difference_update(viable)

        return data

    @staticmethod
    def concat(gdfs: Iterable[GeoDataFrame]) -> GeoDataFrame:
        """Workaround because GeoDataFrame.concat returns DataFrame; we want to preserve CRS."""
        crs = {}

        def generator():
            nonlocal gdfs
            gdfs = iter(gdfs)
            gdf = next(gdfs)
            for col in gdf:
                if not isinstance(gdf[col], GeoSeries):
                    continue
                gs: GeoSeries = gdf[col]
                crs[col] = gs.crs
            yield gdf
            yield from gdfs

        result: DataFrame = pd.concat(generator())
        result: GeoDataFrame = GeoDataFrame({
            col: (
                result[col] if col not in crs
                else GeoSeries(result[col], crs=crs)
            )
            for col in result
        })
        return result


class DescriptorData:
    """
        Allows the user to access the serialized data that has been extracted and processed from the raw data source.
        Source.data.from_file() -> GeoDataFrame
        source().data           -> GeoDataFrame
    """
    _cache_data: CacheData[object, GeoDataFrame] = CacheData()

    def __get__(self, instance, owner) -> Union[GeoDataFrame, 'DescriptorData']:
        if instance in self._cache_data:
            return self._cache_data[instance]
        from ValidateOSM.source import Source
        self._instance: Source = instance
        self._owner: Type[Source] = owner
        if instance is not None:
            if instance.raw is None:
                try:
                    return self._cache_data.setdefault(self._instance, self.from_file())
                except FileNotFoundError as e:
                    raise FileNotFoundError(
                        f"Attempted to call {self._instance}.data when self.instance.raw is None and "
                        f"the result has not yet been serialized."
                    ) from e
            else:
                data = self._cache_data[instance]
                data.to_feather(self.path)
                del instance.raw
                return data
        return self

    def from_file(self) -> GeoDataFrame:
        return gpd.read_feather(self.path)

    def __delete__(self, instance: object):
        del self._cache_data[instance.__class__]

    def __bool__(self):
        return self._instance in self._cache_data

    @property
    def validating(self) -> set[str]:
        structs = self._cache_data._cache_structs[self._owner]
        return {
            name
            for name, struct in structs.items()
            if struct.subtype is DecoratorData.validate
        }

    @property
    def identifier(self) -> set[str]:
        structs = self._cache_data._cache_structs[self._owner]
        return {
            name
            for name, struct in structs.items()
            if struct.subtype is DecoratorData.identifier
        }

    @property
    def path(self) -> Path:
        cls = self._owner
        path = Path(inspect.getfile(cls)).parent / 'data'
        if not path.exists():
            path.mkdir()
        path = path / f"{cls.__name__}.feather"
        return path

    def delete_file(self):
        os.remove(self.path)
