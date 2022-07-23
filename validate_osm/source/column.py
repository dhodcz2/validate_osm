import dataclasses
import functools
import inspect
from collections.abc import Iterable
from dataclasses import dataclass, field
from functools import cached_property
from typing import Callable, Any, Iterator, Type, Collection, Optional

import numpy as np
import pandas as pd
from geopandas import GeoSeries
from pandas import Series

if False:
    from .source import Source


@dataclass
class StructInheritance:
    name: str
    source: str

    crs: Any
    dependent: Collection[str]
    abstract: bool

    data: Callable[['Source'], Iterable | None] = field(repr=False)
    aggregate: Optional[Callable[['Source'], Iterable | None]] = field(repr=False, default=None)


class DescriptorColumns:
    def __get__(self, instance: 'Source', owner) -> dict[str, StructInheritance] | 'DescriptorColumns':
        if owner is None:
            return self
        if owner not in self._columns:
            setattr(self, '_columns', self.__columns(owner))
        return self._columns[owner]

    def __init__(self):
        self._columns: dict[Type['Source'], dict[str, StructInheritance]] = {}

    def __columns(self, item: Type['Source']) -> dict[str, StructInheritance]:
        from validate_osm.source.source import Source
        if item in self._columns:
            return self._columns[item]
        list_candidates: list[dict] = [
            getattr(source, '_columns')
            for source in item.mro() if
            issubclass(source, Source)
            and hasattr(source, '_columns')
        ]
        names = set(map(dict.keys, list_candidates))
        it_candidates = map(lambda name: (
            candidates[name]
            for candidates in list_candidates
            if name in candidates
        ), names)
        # Recurse from children to parents
        inheritances = map(self.__recursion, it_candidates)
        inheritances = list(map(dataclasses.replace, inheritances))
        for inheritance in inheritances:
            inheritance.data = self._data(inheritance.data)
            inheritance.aggregate = self._aggregate(inheritance.aggregate)

        return {
            inherit.name: inherit
            for inherit in inheritances
        }

    def __recursion(self, candidates: Iterator[StructInheritance]) -> StructInheritance:
        child = dataclasses.replace(next(candidates))
        data = child.data
        data = data is not None and not getattr(data, '__isabstractmethod__', False)
        aggregate = child.aggregate
        aggregate = aggregate is not None and not getattr(aggregate, '__isabstractmethod__', False)

        if not (data and aggregate):
            try:
                parent = self.__recursion(candidates)
            except StopIteration:
                ...
            else:
                if not data:
                    child.data = parent.data
                if not aggregate:
                    child.aggregate = parent.aggregate

        return child

    def __data(self, func) -> Callable[['Source'], Iterable | None]:
        # Decorate a column's data method
        def wrapper(source: 'Source'):
            if self.abstract:
                raise TypeError(f'{self} is an abstract method.')

            data: Iterable = func(source)

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
                elif isinstance(data, dict):
                    data = pd.Series(data, dtype=self.dtype)

                else:
                    data = np.ndarray(data, dtype=self.dtype)

            return data

        return wrapper

    def __aggregate(self, func) -> Optional[Callable[['Source'], Iterable | None]]:
        # Decorate a column's aggregate method
        def wrapper(source: 'Source'):
            return func(source)

        return wrapper


class DecoratorColumn:
    def __init__(self, dtype, crs=None, dependent=None):
        if dependent is None:
            self.dependent = set()
        elif isinstance(dependent, str):
            self.dependent = {dependent}
        else:
            self.dependent = set(dependent)

        self.dtype = dtype
        self.crs = crs

    @property
    def frame(self):
        frame = inspect.currentframe()
        frames = inspect.getouterframes(frame)
        return frames[1]

    def __call__(self, func: Callable[['Source'], Iterable | None]):
        frame = self.frame
        _columns: dict[str, StructInheritance] = frame.frame.f_locals.setdefault('_columns', {})
        argspec = inspect.getfullargspec(func)
        if argspec.args != self.args:
            raise SyntaxError(
                f"{func.__name__} has argspec of {argspec}; {self.__class__.__name__} expects {self.args}"
            )
        name = func.__name__
        self.struct = _columns[name] = StructInheritance(
            name=name,
            source=frame.frame.f_locals['__qualname__'],
            crs=self.crs,
            dependent=self.dependent,
            data=func,
            aggregate=None,
            abstract=getattr(func, '__isabstractmethod__', False),
        )
        return func

    def aggregate(self, func: Callable[['Source'], Iterable | None]):
        self.struct._aggregate = func
        return self.struct.data
