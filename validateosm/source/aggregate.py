import dataclasses
import inspect
import warnings
from collections import UserDict
from typing import Callable, Iterator, Union, Any, Type
from typing import Optional
from weakref import WeakKeyDictionary

import pandas
from geopandas import GeoDataFrame, GeoSeries
from pandas import Series

from validateosm.source.pipe import DescriptorPipe


@dataclasses.dataclass
class StructAggregate:
    name: str
    crs: Any
    dtype: Any
    dependent: Optional[set[str]]
    func: Callable

    def __post_init__(self):
        if isinstance(self.dependent, str):
            self.dependent = {self.dependent}
        elif self.dependent is None:
            pass
        elif not isinstance(self.dependent, set):
            self.dependent = set(self.dependent)
        else:
            raise TypeError(self.dependent)

    def __hash__(self):
        return hash(self.name)

    @property
    def decorated_func(self) -> Callable:
        def wrapper(source: type):
            obj: object = self.func(source)
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


class DecoratorAggregate:
    def __init__(self, name: str, dtype=None, crs=None, dependent: Union[str, set[str]] = None):
        self.name = name
        self.dtype = dtype
        self.crs = crs
        self.dependent = (
            {dependent} if isinstance(dependent, str)
            else dependent if dependent is None or isinstance(dependent, set)
            else set(dependent)
        )

    def __call__(self, func: Callable):
        frame = inspect.currentframe()
        frames = inspect.getouterframes(frame)
        frame = frames[1]
        _aggregate: dict[str, StructAggregate] = frame.frame.f_locals.setdefault('_aggregate', {})
        _aggregate[self.name] = StructAggregate(
            name=self.name,
            dtype=self.dtype,
            crs=self.crs,
            dependent=self.dependent,
            func=func
        )
        return func


class CacheStructs(UserDict):
    def __init__(self):
        super(CacheStructs, self).__init__()
        self.data: WeakKeyDictionary[object, StructAggregate] = WeakKeyDictionary()

    def __missing__(self, source: type):
        from validateosm.source import Source
        sources: Iterator[Type[Source]] = (s for s in source.mro()[::-1] if issubclass(s, Source))
        sources = list(sources)
        structs: dict[str: StructAggregate] = getattr(source, '_aggregate')
        structs.update({
            name: struct
            for source in sources
            for name, struct in getattr(source, '_aggregate', {}).items()
        })
        self.data[source] = structs
        return structs


class CacheAggregate(UserDict):
    _cache: WeakKeyDictionary[type, dict[str, StructAggregate]] = CacheStructs()

    def __init__(self):
        super(CacheAggregate, self).__init__()
        self.data: WeakKeyDictionary[object, GeoDataFrame] = WeakKeyDictionary()

    def __missing__(self, source: object):
        data = self.data[source] = self.resolve(source)
        return data

    def resolve(self, source: object):
        structs = self._cache[source.__class__]
        indie = {
            name: struct.decorated_func(source)
            for name, struct in structs.items()
            if not struct.dependent
        }
        rows = max(len(s) for s in indie.values())
        aggregate = GeoDataFrame({
            name: (
                series.repeat(rows).reset_index(drop=True)
                if len(series) == 1
                else series
            )
            for name, series in indie.items()
        })
        self.data[source] = aggregate
        depend: set[StructAggregate] = {
            struct
            for struct in structs.values()
            if struct.dependent
        }
        while depend:
            viable: set[StructAggregate] = {
                struct for struct in depend if not struct.dependent.difference(aggregate.columns)
            }
            if not viable:
                raise RuntimeError(f"Cannot resolve with cross-dependencies: {depend}")
            for struct in viable:
                series = struct.func(source)
                aggregate[struct.name] = series.repeat(rows) if len(series) == 1 else series
            depend = {struct for struct in depend if struct not in viable}
        aggregate.index.name = 'i'
        return aggregate


class DescriptorAggregate(DescriptorPipe):
    _cache: CacheAggregate[object, GeoDataFrame] = CacheAggregate()

    def __get__(self, instance, owner) -> Union[GeoDataFrame, 'DescriptorAggregate']:
        deldata = instance is not None and instance not in self._cache
        ret = super(DescriptorAggregate, self).__get__(instance, owner)
        if deldata:
            del instance.data
        return ret
