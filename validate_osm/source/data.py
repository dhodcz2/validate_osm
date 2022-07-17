import inspect
import logging
from collections.abc import Iterable
from dataclasses import dataclass
from typing import Callable, Any, Iterator, Type, Collection

import numpy as np
import pandas as pd
from geopandas import GeoDataFrame
from geopandas import GeoSeries
from pandas import Series

from .logger import logged_subprocess

if False:
    from .source import Source


@dataclass
class StructInheritance:
    name: str
    cls: str
    func: Callable
    dtype: str
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
    def decorated(self) -> Callable[['Source'], Series | np.ndarray]:
        if self.abstract:
            raise TypeError(f'{self} is an abstract method.')

        def wrapper(source: 'Source'):
            data: Iterable = self.func(source)

            if self.dtype == 'geometry':
                if isinstance(data, GeoSeries):
                    if data.crs is not self.crs:
                        data = data.to_crs(self.crs)
                elif self.crs is None:
                    data = GeoSeries(data, crs=None)
                else:
                    raise TypeError(f'Expected GeoSeries, got {type(data)}.')

            else:
                if isinstance(data, (Series, np.ndarray)):
                    if data.dtype != self.dtype:
                        data = data.astype(self.dtype)
                else:
                    data = np.ndarray(data, dtype=self.dtype)

            return data

        return wrapper


class DescriptorInheritances:
    def __init__(self):
        self.data: dict[Type['Source'], dict[str, StructInheritance]] = {}

    def __getitem__(self, item: Type['Source']) -> dict[str, 'StructInheritance']:
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

        def inherit(name: str) -> StructInheritance:
            if hasattr(item, '_data') and name in (structs := getattr(item, '_data', {})):
                return structs[name]
            struct_list: Iterator[StructInheritance] = [
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

            return StructInheritance(
                name=name,
                func=func,
                dtype=dtype,
                abstract=abstract,
                crs=crs,
                cls=cls,
                dependent=dependent
            )

        result = self.data[item] = {name: inherit(name) for name in names}
        return result


class DecoratorData:
    """
    @DecoratorData(dtype='datetime64[ns]', dependent='geometry', ...)
    """

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
        _data: dict[str, StructInheritance] = frame.frame.f_locals.setdefault('_data', {})
        name = func.__name__
        _data[name] = StructInheritance(
            name=func.__name__,
            func=func,
            cls=frame.frame.f_locals['__qualname__'],
            dtype=self.dtype,
            abstract=getattr(func, '__isabstractmethod__', False),
            crs=self.crs,
            dependent=self.dependent
        )
        return func


class DescriptorData:
    # TODO: How do we load the data files instead of processing it again?
    inherit = DescriptorInheritances()

    def __get__(self, instance: 'Source', owner: Type['Source']) -> GeoDataFrame | 'DescriptorData':
        self.source = instance
        self.Source = owner
        if instance is None:
            return self
        if instance.resource:
            self.__set__(instance, self.__data())
        if hasattr(instance, '_data'):
            return instance._data
        raise RecursionError

    def __set__(self, instance, value):
        if instance is None:
            raise ValueError(instance)
        self.cache[instance] = value

    def __delete__(self, instance):
        if hasattr(instance, '_data'):
            del instance._data

    def __data(self) -> GeoDataFrame:
        with logged_subprocess(f'transforming {self.source.name} resource to data', level=logging.INFO):
            inherit = self.inherit[self.Source]

            indie = {
                col: struct.decorated(self.source)
                for col, struct in inherit.items()
                if not struct.dependent
            }
            repeat = max(map(len, indie.values()))
            data = GeoDataFrame({
                key: self.__repeat(value, repeat)
                for key, value in indie.items()
            })
            self.__set__(self.source, data)

            depend: set[StructInheritance] = {
                struct
                for struct in inherit.values()
                if struct.dependent
            }

            while depend:
                viable = {
                    struct
                    for struct in depend
                    if not struct.dependent.difference(data.columns)
                }
                if not viable:
                    raise RuntimeError(f'Cannot resolve cross-dependencies for {depend}')

                for struct in viable:
                    column = struct.decorated(self.source)
                    data[struct.name] = self.__repeat(column, repeat)

            # ensure that geometry is valid
            geometry: GeoSeries = data.geometry
            data.geometry.update(geometry[~geometry.is_valid].buffer(0))

            # assign the resource name as a categorical column
            data['name'] = pd.Categorical([self.source.name] * len(data))

            return data

    @staticmethod
    def __repeat(col: Collection, repeat: int):
        if isinstance(col, Series):
            return col.repeat(repeat).reset_index(drop=True)
        elif isinstance(col, np.ndarray):
            return col.repeat(repeat)
        else:
            raise NotImplementedError(type(col))
