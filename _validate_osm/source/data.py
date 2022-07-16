import dataclasses
import inspect
import logging
import os
from pathlib import Path
from typing import Callable, Any, Iterator, Union, Type
from weakref import WeakKeyDictionary

import pandas
import pandas as pd
from geopandas import GeoDataFrame
from geopandas import GeoSeries
from pandas import Series
from pandas.core.indexes.range import RangeIndex

from validate_osm.logger import logged_subprocess, logger
from validate_osm.source.bbox import BBox
from validate_osm.source.resource_ import StructFile, StructFiles
from validate_osm.util import concat

if False | False:
    from validate_osm.source.source import Source


class DecoratorData:
    def __init__(self, dtype, crs=None, dependent=None):
        if dependent is None:
            dependent = set()
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
        def wrapper(source: 'Source'):
            logger.debug(f'{self.name}')
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
        self.data: dict[Type['Source'], dict[str, StructData]] = {}

    def __getitem__(self, item: Type['Source']) -> dict[str, StructData]:
        if item in self.data:
            return self.data[item]
        from validate_osm.source.source import Source
        sources = [
            s for s in item.mro()[:0:-1]
            if issubclass(s, Source)
        ]
        # start reversed to maintain O(x) instead of O(x^2)
        inheritances = [
            self[source]
            for source in sources
        ]
        inheritances.reverse()
        names: set[str] = set(getattr(item, '_data', {}).keys())
        names.update(
            key
            for inherit in inheritances
            for key in inherit.keys()
        )

        def inherit(name: str) -> StructData:
            if hasattr(item, '_data') and name in (structs := getattr(item, '_data', {})):
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

            if name in item.__dict__:
                func = getattr(item, name)
                abstract = getattr(func, '__isabstractmethod__', False)
                if abstract:
                    raise RuntimeError(f"{item=}; {name=}, {abstract=}")
                cls = item.__name__
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

        result = self.data[item] = {name: inherit(name) for name in names}
        return result


class DescriptorData:
    cache: WeakKeyDictionary[object, GeoDataFrame] = WeakKeyDictionary()
    structs: dict[type, dict[str, StructData]] = DescriptorStructs()

    def __getitem__(self, item: BBox) -> list[StructFile, StructFiles]:
        return self.owner.resource.__getitem__(item)

    def __get__(self, instance: 'Source', owner: Type['Source']) -> Union[GeoDataFrame, 'DescriptorData']:
        if instance in self.cache:
            return self.cache[instance]
        self.source = instance
        self.owner = owner

        # Case: Source.__class__.data
        if instance is None:
            return self

        # Case: Compare.source.data
        if instance.compare is not None:
            self.__set__(instance, self.transform)

        # Case: Source.data
        if instance.compare is None:
            self.__set__(instance, concat(self))

        return self.__get__(instance, owner)

    @property
    def transform(self) -> GeoDataFrame:
        with logged_subprocess(f'transforming {self.source.name} resource to data', level=logging.INFO):
            structs = self.structs[self.owner]

            indie = {
                name: struct.decorated_func(self.source)
                for name, struct in structs.items()
                if not struct.dependent
            }
            rows = max(len(obj) for obj in indie.values())
            data = GeoDataFrame({
                name: (
                    series._repeats(rows).reset_index(drop=True)
                    if len(series) == 1
                    else series
                )
                for name, series in indie.items()
            })

            # Extract data that is dependent on data that has been extracted.
            self.cache[self.source] = data
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
                    series = struct.decorated_func(self.source)
                    data[struct.name] = (
                        series.repeat(rows).reset_index(drop=True)
                        if len(series) == 1
                        else series
                    )
                    # data[struct.name] = series.repeat(rows) if len(series) == 1 else series
                depend.difference_update(viable)

            geometry: GeoSeries = data.geometry
            data.geometry.update(geometry[~geometry.is_valid].buffer(0))

            return data

    def __delete__(self, instance):
        if instance in self.cache:
            del self.cache[instance]

    def __hash__(self):
        return hash(self.source)

    def __eq__(self, other):
        return self.source == other

    def __set__(self, instance, value: GeoDataFrame):
        if instance is None:
            raise ValueError(instance)
        value = value.assign(iloc=pd.Series(range(len(value)), dtype='int32', index=value.index))
        self.cache[instance] = value

    @classmethod
    @property
    def directory(cls):
        return Path(inspect.getfile(cls)).parent / 'static' / 'source' / cls.__name__
