from validate_osm.source.resource_ import DescriptorStatic
import geopandas as gpd
import os
from pathlib import Path

from validate_osm.logger import logger, logged_subprocess
import dataclasses
from validate_osm.logger import logged_subprocess, logger
import inspect
from collections import UserDict
from validate_osm.source.resource_ import StructFile, StructFiles
from typing import Callable, Any, Iterator, Union, Type
from weakref import WeakKeyDictionary

import pandas
import pandas as pd
from geopandas import GeoDataFrame
from geopandas import GeoSeries
from pandas import Series
from pandas.core.indexes.range import RangeIndex

from validate_osm.source._pipe import DescriptorPipeSerialize


class DecoratorData:
    def __init__(self, dtype, crs=None, dependent: Union[str, set[str]] = set()):
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
        _data: dict[str, StructData] = frame.frame.f_locals.setdefault('_data', {})
        name = func.__name__
        _data[name] = StructData(
            name=func.__name__,
            func=func,
            cls=frame.frame.f_locals['__qualname__'],
            dtype=self.dtype,
            subtype=self.__class__,
            abstract=getattr(func, '__isabstractmethod__', False),
            crs=self.crs,
            dependent=self.dependent
        )
        return func


@dataclasses.dataclass
class StructData:
    name: str
    cls: str
    func: Callable
    dtype: str
    subtype: type
    abstract: bool
    crs: Any
    dependent: set[str]

    def __hash__(self):
        return hash(self.name)

    def __eq__(self, other):
        return self.name == other

    def __repr__(self):
        return f"{self.__class__.__name__}[{self.name} from {self.cls}]"

    @property
    def decorated_func(self):
        def wrapper(source: object):
            if self.abstract:
                raise TypeError(f"{self.__repr__()} is abstract method.")
            obj: object = self.func(source)
            # Expect a GeoSeries
            if self.dtype == 'geometry':
                if self.crs is None:
                    obj = GeoSeries(obj, crs=None)
                else:
                    if not isinstance(obj, GeoSeries):
                        raise TypeError(
                            f"{self.func.__qualname__} must be returned as a GeoSeries so that it may specify a CRS."
                        )
                    if obj.crs is None:
                        raise ValueError(f"{obj} must specify a CRS.")
                    obj = obj.to_crs(self.crs)

            # Got a Series
            if isinstance(obj, Series):
                if not isinstance(obj.index, pandas.core.indexes.range.RangeIndex):
                    # source.logger.debug(
                    #     f"index returns a {type(obj.index)}; naively passing this may result in a mismatched "
                    #     f"column index. Resetting this index so that column indices align regardless of implementation."
                    # )
                    obj = obj.reset_index(drop=True)
                if self.dtype is not None:
                    obj = obj.astype(self.dtype)
            else:
                obj = Series(obj, dtype=self.dtype)
            return obj

        return wrapper


class DescriptorStructs:
    def __init__(self):
        self.data: dict[type, dict[str, StructData]] = {}

    def __getitem__(self, item):
        if item in self.data:
            return self.data[item]
        from validate_osm.source.source import Source
        source: Type[Source] = item
        sources = [
            s for s in source.mro()[:0:-1]
            if issubclass(s, Source)
        ]
        # start reversed to maintain O(x) instead of O(x^2)
        inheritances = [
            self[source]
            for source in sources
        ]
        inheritances.reverse()
        names: set[str] = set(getattr(source, '_data', {}).keys())
        names.update(
            key
            for inherit in inheritances
            for key in inherit.keys()
        )

        def inherit(name: str) -> StructData:
            if hasattr(source, '_data') and name in (structs := getattr(source, '_data', {})):
                return structs[name]
            struct_list: Iterator[StructData] = [
                inherit[name]
                for inherit in inheritances
                if name in inherit
            ]
            structs = iter(struct_list)
            try:
                struct = next(structs)
            except StopIteration as e:
                raise RuntimeError from e

            # Get first inheritance
            dtype = struct.dtype
            subtype = struct.subtype
            crs = struct.crs
            dependent = struct.dependent

            if name in source.__dict__:
                func = getattr(source, name)
                abstract = getattr(func, '__isabstractmethod__', False)
                if abstract:
                    raise RuntimeError(f"{source=}; {name=}, {abstract=}")
                cls = source.__name__
            else:
                # Get first inheritance that isn't abstract
                if struct.abstract:
                    for struct in structs:
                        if not struct.abstract:
                            break
                abstract = struct.abstract
                func = struct.func
                cls = struct.cls

            return StructData(
                name=name,
                func=func,
                dtype=dtype,
                subtype=subtype,
                abstract=abstract,
                crs=crs,
                cls=cls,
                dependent=dependent
            )

        result = self.data[source] = {name: inherit(name) for name in names}
        return result


class DescriptorData:
    cache: WeakKeyDictionary[object, GeoDataFrame] = WeakKeyDictionary()
    structs = DescriptorStructs()

    # TODO: If we call just Source.data, it will load

    def __get__(self, instance, owner) -> Union['DescriptorData', GeoDataFrame]:
        if instance in self.cache:
            return self.cache[instance]
        self._instance = instance
        self._owner = owner
        if instance is None:
            return self
        from validate_osm.source.source import Source
        source: Source = instance
        owner: Type[Source]
        path = self.path
        if not source.redo and path.exists():
            with logged_subprocess(f'reading {owner.__name__}.data from {path} ({StructFile.size(path)})'):
                self.__set__(instance, gpd.read_feather(path))
        else:
            with logged_subprocess(f'building {owner.__name__}.data'), self as data:
                self.__set__(instance, data)
        return self.__get__(instance, owner)

    def __set__(self, instance, value: GeoDataFrame):
        from validate_osm.source.source import Source
        instance: Source
        value = value.assign(iloc=pd.Series(range(len(value)), dtype='int32', index=value.index))
        self.cache[instance] = value

    def __enter__(self):
        from validate_osm.source.source import Source
        source: Source = self._instance
        structs = self.structs[self._owner]

        indie = {
            name: struct.decorated_func(source)
            for name, struct in structs.items()
            if not struct.dependent
        }
        rows = max(len(obj) for obj in indie.values())
        data = GeoDataFrame({
            name: (
                series.repeat(rows).reset_index(drop=True)
                if len(series) == 1
                else series
            )
            for name, series in indie.items()
        })

        # Extract data that is dependent on data that has been extracted.
        self.cache[source] = data
        depend: set[StructData] = {
            struct
            for struct in structs.values()
            if struct.dependent
        }
        while depend:
            viable: set[StructData] = {
                struct for struct in depend
                if not struct.dependent.difference(data.columns)  # If there is no dif, then all the req are avail
            }
            if not viable:
                raise RuntimeError(f"Cannot resolve with cross-dependencies: {depend}")
            for struct in viable:
                series = struct.decorated_func(source)
                data[struct.name] = (
                    series.repeat(rows).reset_index(drop=True)
                    if len(series) == 1
                    else series
                )
                # data[struct.name] = series.repeat(rows) if len(series) == 1 else series
            depend.difference_update(viable)
        # The reason this doesn't just assign to data['ubid'] is because SourceOSM.group() wants to drop rows
        data = source.group()
        # Problem is, the geometry has been processed into 3857
        bbox = source.bbox.to_crs(data.crs).ellipsoidal
        data = data[data.intersects(bbox)]
        return data

    def __exit__(self, exc_type, exc_val, exc_tb):
        from validate_osm.source.source import Source
        source: Source = self._instance
        if exc_type is None and source.serialize:
            path = self.path
            if not path.parent.exists():
                os.makedirs(path.parent)
            self.cache[self._instance].to_feather(self.path)

    def __delete__(self, instance):
        if instance in self.cache:
            del self.cache[instance]

    @property
    def directory(self) -> Path:
        from validate_osm.source.source import Source
        source: Type[Source] = self._owner
        return source.directory / 'source'


class DescriptorData:
    cache: WeakKeyDictionary[object, GeoDataFrame] = WeakKeyDictionary()
    structs = DescriptorStructs()

    def __get__(self, instance, owner) -> Union['DescriptorDatao', GeoDataFrame]:
        if instance in self.cache:
            return self.cache[instance]
        self.instance = instance
        self.owner = owner
        if instance is None:
            return self

        from validate_osm.source.source import Source
        instance: Source
        owner: Type[Source]

        if issubclass(owner.resource.__class__, DescriptorStatic):
            ...
        elif isinstance(owner.resource, )


    # TODO: For each File that doesn't have File.source, load it into source.resource, building source.data, serialize
    #   and then concat all source.datas

    # TODO: Most optimized file storage
    """
    static  resource    ocm      illinois.feather
    static  resource    msbf     17031.feather
    static  source      ocm       illinois.feather
    static  source      ocm      17031.feather
    """
