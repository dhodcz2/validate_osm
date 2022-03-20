import dataclasses
import functools
import inspect
import itertools
import weakref
from collections import UserDict
from typing import Callable, Collection, Generator
from weakref import WeakKeyDictionary

import networkx
import pandas as pd
from geopandas import GeoDataFrame
from networkx.algorithms.components.connected import connected_components

from validate_osm.source.pipe import DescriptorPipe


@dataclasses.dataclass
class Groups:
    data: GeoDataFrame = dataclasses.field(repr=False)
    grouped: Collection[Collection[int]]
    _grouped: Collection[Collection[int]] = dataclasses.field(init=False, repr=False)
    ungrouped: Collection[int]
    _ungrouped: Collection[int] = dataclasses.field(init=False, repr=False)

    @property
    def grouped(self) -> list[GeoDataFrame]:
        return [
            self.data.iloc[group]
            for group in self._grouped
        ]

    @grouped.setter
    def grouped(self, value):
        self._grouped = [
            list(v) for v in value
        ]

    @property
    def ungrouped(self) -> GeoDataFrame:
        return self.data.iloc[self._ungrouped]

    @ungrouped.setter
    def ungrouped(self, value):
        self._ungrouped = list(value)

    # @functools.cached_property
    # def index_grouped(self) -> list[tuple]:
    #     index = self.data.index
    #     members = (
    #         group[0]
    #         for group in self._grouped
    #     )
    #     return [
    #         index[member][:-2]
    #         for member in members
    #     ]
    #
    # @functools.cached_property
    # def index_ungrouped(self) -> list[tuple]:
    #     return [
    #         tuple[:2]
    #         for tuple in self.data.iloc[self._ungrouped].index
    #
    #     ]

    @functools.cached_property
    def index(self):
        # index_ungrouped = [
        #     tuple[:2]
        #     for tuple in self.data.iloc[self._ungrouped].index
        # ]
        # index = self.data.index
        # members = (
        #     group[0]
        #     for group in self._grouped
        # )
        # index_grouped = [
        #     index[member][:-2]
        #     for member in members
        # ]
        # return pd.MultiIndex.from_tuples(
        #     itertools.chain(index_ungrouped, index_grouped),
        #     names=self.data.index.names[:2]
        # )
        ungrouped: pd.MultiIndex = self.data.iloc[self._ungrouped].index
        index_ungrouped = zip(ungrouped.get_level_values('ubid'), ungrouped.get_level_values('name'))
        # grouped = self.grouped
        # first = [grouped.iloc[0] for group in grouped]
        first_indices = [
            df.index[0]
            for df in self.grouped
        ]
        index_grouped = zip((idx[0] for idx in first_indices), (idx[1] for idx in first_indices))
        # grouped: pd.MultiIndex = self.data.iloc[itertools.chain.from_iterable(self._grouped)].index
        # index_grouped = zip(grouped.get_level_values('ubid'), grouped.get_level_values('name'))
        return pd.MultiIndex.from_tuples(itertools.chain(index_ungrouped, index_grouped), names=['ubid', 'name'])


@dataclasses.dataclass
class StructGroup:
    name: str
    func: Callable
    dependent: bool

    def __hash__(self):
        return hash(self.name)

    def __eq__(self, other):
        return self.name == other


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
        frame.frame.f_locals.setdefault('_group', {})[self.name] = StructGroup(
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
        self.data: WeakKeyDictionary[type, set[StructGroup]] = WeakKeyDictionary()

    def __missing__(self, source: type):
        from validate_osm.source import Source
        structs: set[StructGroup] = {
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
    _cache: CacheStructs[type, set[StructGroup]] = CacheStructs()

    def __init__(self):
        super(CacheGroups, self).__init__()
        self.data: WeakKeyDictionary[object, Groups] = WeakKeyDictionary()
        self.structs: CacheStructs[type, set[StructGroup]] = CacheStructs()

    def __missing__(self, source: object) -> Groups:
        from validate_osm.source import Source
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
    _cache: WeakKeyDictionary[object, Groups] = CacheGroups()

    def __get__(self, instance, owner) -> Groups:
        return super(DescriptorGroup, self).__get__(instance, owner)

    def __set__(self, instance, value):
        self._cache[instance] = value



