from annoy import AnnoyIndex
import geopandas as gpd

import dataclasses
import functools
import inspect
from collections import UserDict
from typing import Callable, Collection, Generator, Type, Iterable
from weakref import WeakKeyDictionary

import networkx
from geopandas import GeoDataFrame
from networkx.algorithms.components.connected import connected_components
from validateosm.source.groups import  Groups, CacheGroups, CacheStructs


from validateosm.source.pipe import DescriptorPipe
from validateosm.source.source import Source

# TODO: How to group using footprint?


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

    def __len__(self) -> int:
        return len(self._grouped) + len(self._ungrouped)

class CacheGroups(UserDict):
    def __init__(self):
        super(CacheGroups, self).__init__()
        self.data: WeakKeyDictionary[object, Groups] = WeakKeyDictionary()
        self.structs: CacheStructs[type, set[]]


class DescriptorGroup:
    # TODO: When del data, also del groups
    def __init__(self):
        self._cache = None
        self.structs: CacheStructs[type, set[StructGroup]] = CacheStructs()
        self.data: dict[str, Groups] = {}

    def __get__(self, instance, owner):
        from validateosm.compare.compare import  Compare
        self._instance: Compare = instance
        self._owner: Type[Compare] = owner
        if instance is None:
            return self

    def __del__(self):
        self.data.clear()

    def __getitem__(self, item: str) -> Groups:
        return self.data[item]

    def __missing__(self, key: str) -> Groups:
        from validateosm.source import Source
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
        footprint = self._instance.footprint(key)
        G.add_nodes_from(footprint)
        G.add_edges_from(footprint)
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


    def __iter__(self) -> Groups:
        data: GeoDataFrame = self._instance.data
        gdfs: Iterable[GeoDataFrame] = (
            data.loc[loc]
            for loc in self._instance.data.groupby('name').groups.values()
        )










