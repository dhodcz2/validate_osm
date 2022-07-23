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

    def aggregate(self):
        # TODO
        raise NotImplementedError()


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


column = DecoratorData
