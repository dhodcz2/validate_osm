import inspect
from ValidateOSM.source.pipe import DescriptorPipe
import sqlite3
import itertools
from collections import UserDict

import functools
from pathlib import Path

import numpy as np
import pandas as pd
from geopandas import GeoDataFrame
import dataclasses
from typing import Iterable, Callable, Optional, Type, Collection, Iterator, Generator
from weakref import WeakKeyDictionary
import networkx
from networkx.algorithms.components.connected import connected_components
import weakref


@dataclasses.dataclass
class Groups:
    _data: GeoDataFrame
    _grouped: Collection[Collection[int]]
    _ungrouped: Collection[int]

    @functools.cached_property
    def grouped(self) -> Generator[GeoDataFrame, None, None]:
        yield from (
            self._data.iloc[group]
            for group in self._grouped
        )

    @functools.cached_property
    def ungrouped(self) -> GeoDataFrame:
        return self._data.iloc[self._ungrouped]

class DecoratorGroup:
    """
    for n in N:
        if all(group[n] is NA for group in groups):
            exclude n
        else
            include n

    thus:
        if n is grouped in a len 1 array:
            it is included in aggregate
    """

    @dataclasses.dataclass
    class Struct:
        name: str
        func: Callable
        dependent: bool

        def __hash__(self):
            return hash(self.name)

        def __eq__(self, other):
            return self.name == other

    def __init__(self, name: str, dependent: str = None):
        self.name = name
        self.dependent = (
            {dependent} if isinstance(dependent, str)
            else dependent if dependent is None or isinstance(dependent, set)
            else set(dependent)
        )

    def __call__(self, func: Callable) -> Callable:
        argspec = inspect.getfullargspec(func)
        if argspec.args != ['self']:
            # TODO: Perhaps list the file and line as well
            raise SyntaxError(
                f"{func.__name__} has argspec of {argspec.args}; {self.__class__.__name__} expects {['self']}"
            )
        if func.__name__ != '_':
            raise SyntaxError(
                f"{func.__name__} should be defined with a name of _ so that it does not conflict in the namespace"
                f" with columns marked by DecoratorData."
            )
        frame = inspect.currentframe()
        frames = inspect.getouterframes(frame)
        frame = frames[1]
        frame.frame.f_locals.setdefault('_group', {})[self.name] = DecoratorGroup.Struct(
            name=self.name,
            func=func,
            dependent=self.dependent
        )
        return func


class CacheStructs(UserDict):
    """
    Caches the GroupStructs with regards to a class; because this is cheap, it is cached using class, not instance.
    """

    def __init__(self):
        super(CacheStructs, self).__init__()
        self.data: WeakKeyDictionary[type, set[DecoratorGroup.Struct]] = WeakKeyDictionary()

    def __missing__(self, source: type):
        from ValidateOSM.source import Source
        structs: set[DecoratorGroup.Struct] = {
            struct
            for c in source.mro()[::-1]
            if issubclass(c, Source) and hasattr(c, '_group')
            for struct in getattr(c, '_group').values()
        }
        self.data[source] = structs
        return structs


class CacheGroups(UserDict):
    """
    Caches the Groups with regards to a class
    """
    _cache: CacheStructs[type, set[DecoratorGroup.Struct]] = CacheStructs()

    def __init__(self):
        super(CacheGroups, self).__init__()
        self.data: WeakKeyDictionary[object, Groups] = WeakKeyDictionary()
        self.structs: CacheStructs[type, set[DecoratorGroup.Struct]] = CacheStructs()

    def __missing__(self, source: object) -> Groups:
        from ValidateOSM.source import Source
        source: Source
        structs = self.structs[source.__class__]
        groups = (
            group
            for struct in structs
            for group in struct.func(source)
            if len(group) > 1
        )
        G = networkx.Graph()
        for group in groups:
            G.add_nodes_from(group)
            G.add_edges_from(zip(group[:-1], group[1:]))
        groups = connected_components(G)
        ungrouped: list[int] = [
            group[0]
            for group in groups
            if len(group) == 1
        ]
        grouped: list[Collection[int]] = [
            group
            for group in groups
            if len(group) > 1
        ]
        result = self.data[source] = Groups(
            _data=source.data,
            _grouped=grouped,
            _ungrouped=ungrouped
        )
        return result


class DescriptorGroup(DescriptorPipe):
    _cache = CacheGroups()

    def __get__(self, instance, owner) -> Groups:
        return super(DescriptorGroup, self).__get__(instance, owner)

{
    'A': {
        'osm' ...
    },
    'B': {
        'osm': ...
    },
    'AB': {
        'osm': ...,
        'msbf': ...
    }
}
