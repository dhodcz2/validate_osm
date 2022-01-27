import inspect
from collections import UserDict

import functools

import pandas as pd
from geopandas import GeoDataFrame
import dataclasses
from typing import Iterable, Callable, Optional, Type
from weakref import WeakKeyDictionary
import networkx
from networkx.algorithms.components.connected import connected_components
import weakref

class Groups:
    def __init__(self, data: GeoDataFrame, it_groups: Iterable[Iterable[Iterable[int]]]):
        self.data = data
        G = networkx.Graph()
        for groups in it_groups:
            G.add_nodes_from(groups)
            G.add_edges_from(zip(groups[:-1], groups[1:]))
        self._groups = connected_components(G)

    @functools.cached_property
    def multi_len(self) -> list[GeoDataFrame]:
        return [
            self.data.iloc[group]
            for group in self._groups
            if len(group) > 1
        ]

    @functools.cached_property
    def single_len(self) -> GeoDataFrame:
        iloc = (
            group[0]
            for group in self._groups
            if len(group) == 1
        )
        return self.data.iloc[iloc]


class _DecoratorGroup:
    dependent: bool
    expected: list[str]

    @dataclasses.dataclass
    class Struct:
        name: str
        func: Callable
        dependent: bool

        def __hash__(self):
            return hash(self.name)

        def __eq__(self, other):
            return self.name == other

    def __init__(self, name: str):
        self.name = name

    def __call__(self, func: Callable) -> Callable:
        argspec = inspect.getfullargspec(func)
        if argspec.args != self.expected:
            # TODO: Perhaps list the file and line as well
            raise SyntaxError(
                f"{func.__name__} has argspec of {argspec}; {self.__class__.__name__} expects {self.expected}"
            )
        if func.__name__ != '_':
            raise SyntaxError(
                f"{func.__name__} should be defined with a name of _ so that it does not conflict in the namespace"
                f" with columns marked by DecoratorData."
            )
        frame = inspect.currentframe()
        frames = inspect.getouterframes(frame)
        frame = frames[1]
        frame.frame.f_locals.setdefault('_group', {})[self.name] = _DecoratorGroup.Struct(
            name=self.name,
            func=func,
            dependent=self.dependent
        )
        return func


class DecoratorGroup:
    """
    Decorate a function to be used for defining relations
    """
    def __new__(cls, *args, **kwargs):
        raise SyntaxError(f"DecoratorGroup exists as syntactic sugar; wrap with {cls.independent.__name__} or "
                          f"{cls.dependent.__name__}")
    class independent(_DecoratorGroup):
        dependent = False
        expected = ['self']

    class dependent(_DecoratorGroup):
        dependent = True
        expected = ['self', 'dependency']


class CacheStructs(UserDict):
    """
    Caches the GroupStructs with regards to a class; because this is cheap, it is cached using class, not instance.
    """
    def __init__(self):
        super(CacheStructs, self).__init__()
        self.data: WeakKeyDictionary[type, set[_DecoratorGroup.Struct]] = WeakKeyDictionary()
        from ValidateOSM.source.source import  Source
        self._Source = Source

    def __missing__(self, source: type):
        Source = self._Source
        structs = self.data[source] =  {
            struct
            for c in source.mro()[::-1]
            if isinstance(c, Source)
            for struct in getattr(c, '_group')
        }
        return structs


class DescriptorGroup:
    """
    This attribute entails the dependent or independent Group that has been assigned to an instance;
        Groups are relations that represent a many-to-one relation between "data" and "aggregate"
        Independent groups are self-referential
        Dependent groups are built referencing another Source (e.g., what boundary in OSM defines a "building")
    """
    _cache_group: WeakKeyDictionary[object, Groups] = WeakKeyDictionary()
    _cache_structs: WeakKeyDictionary[type, set[_DecoratorGroup.Struct]] = CacheStructs()

    def __init__(self):
        self._instance = None
        self._owner = None

    def __get__(self, instance: None, owner: type):
        self._instance = instance
        self._owner = owner
        return self

    def __bool__(self):
        return self._owner in self._cache_group

    def __call__(self, source: Optional[object]) -> None:
        from ValidateOSM.source import Source
        self._instance: Source
        self._owner: Type[Source]
        source: Source
        if not isinstance(self._instance.aggregate.index, pd.RangeIndex):
            raise ValueError(f"Check to make sure that the index is 0,1,2,3...n")

        if source is None:  # self grouping
            if self._instance is None:
                raise ValueError(f"{source} must be instantiated to perform an independent grouping")
            groupings = {
                struct.name: struct.func(self._instance)
                for struct in self._cache_structs[self._owner]
                if not struct.dependent
            }
            self._cache_group[self._instance] = Groups(self._instance.data, groupings)
            # self._cache_group[self._owner] = Group(self._instance.data, groupings)
        else:
            groupings = {
                struct.name: struct.func(source, self._instance)
                for struct in self._cache_structs[self._owner]
                if struct.dependent
            }
            self._cache_group[source.__class__] = Groups(data=source.data, it_groups=groupings)



    def __delete__(self, instance):
        del self._cache_group[instance]

    @property
    def single_len(self) -> GeoDataFrame:
        f"""
        :return: A single GeoDataFrame of the all the entries in {self.__class__}.data which have not been grouped
            with other entries. This is given as a single GeoDataFrame to make iteration faster.
        """
        try:
            return self._cache_group[self._owner].single_len
        except KeyError as e:
            raise ValueError(
                f"Could not resolve a cached group for {self._instance}. Be sure to call Source.group.__call__"
            ) from e

    @property
    def multi_len(self) -> list[GeoDataFrame]:
        f"""
        :return: A list of GeoDataFrames which have been determined to have a relationship in some way by a 
            {DecoratorGroup} function.
        """
        try:
            return self._cache_group[self._owner].multi_len
        except KeyError as e:
            raise ValueError(
                f"Could not resolve a cached group for {self._instance}. Be sure to call Source.group.__call__"
            ) from e

    def _group(self, instance: object, dependent: bool) -> Groups:
        structs = self._cache_structs[instance.__class__]
        Source = self._Source
        instance: Source
        it_groups = (
            struct.func(instance, self._owner)
            for struct in structs
            if struct.dependent
        ) if dependent else (
            struct.func(instance)
            for struct in structs
            if not struct.dependent
        )
        return Groups(data=instance.data, it_groups=it_groups)

