import pandas.core.indexes.range
from networkx.algorithms.components.connected import connected_components
import networkx
import dataclasses
import functools
import itertools
import warnings
from collections import UserDict
from pathlib import Path
from weakref import WeakKeyDictionary

import shapely.geometry
from annoy import AnnoyIndex
from typing import Generator, Union, Optional, Type, Iterable, Callable, Collection
import geopandas as gpd
import pandas as pd
from geopandas import GeoDataFrame, GeoSeries
from pandas import Series

from validateosm.util import concat
from validateosm.source.aggregate import (
    StructAggregate as StructSingleAggregate,
    CacheStructs as CacheSingleStructs
)
from validateosm.source.source import Source
from validateosm.source.groups import Groups
from typing import Iterator


@dataclasses.dataclass
class StructAggregate(StructSingleAggregate):

    @property
    def decorated_func(self) -> Callable:
        def wrapper(groups: Iterable[Groups]) -> Callable[[Iterable[Groups]], Union[Series, GeoSeries]]:
            def gen() -> Generator[object, None, None]:
                for group in groups:
                    result = self.func(groups)
                    if self.dtype == 'geometry' and self.crs is not None:
                        if not isinstance(result, GeoSeries):
                            raise TypeError(f"{result} must be returned as a GeoSeries so that it may specify a CRS.")
                        if result.crs is None:
                            raise ValueError(f"{result} must specify a CRS.")
                        result = result.to_crs(self.crs)

                    if isinstance(result, Series):
                        if not isinstance(result.index, pandas.core.indexes.range.RangeIndex):
                            warnings.warn(
                                f"{result}.index returns a {type(result.index)}; naively passing this may result in a mismatched "
                                f"column index. Resetting this index so that column indices align regardless of implementation."
                            )
                            result = result.reset_index(drop=True)

                    try:
                        yield from result
                    except TypeError:
                        yield from itertools.repeat(result, len(group))

            if self.dtype == 'geometry':
                result = GeoSeries(gen(), crs=self.crs)
            else:
                result = Series(gen())
                result = result.astype(self.dtype)

            return result

        return wrapper


class CacheStructs(UserDict):
    def __init__(self):
        super(CacheStructs, self).__init__()
        self.data: WeakKeyDictionary[str, StructAggregate] = WeakKeyDictionary()

    def __get__(self, instance: 'DescriptorAggregate', owner: Type['DescriptorAggregate']):
        self._instance = instance
        self._owner = owner


# class DescriptorGroups:
#     def __init__(self):
#         self.structs = CacheStructs()
#
#     def __get__(self, instance, owner) -> list[Groups]:
#         self._instance: DescriptorAggregate = instance
#         self._owner: Type[DescriptorAggregate] = owner
#         if instance is None:
#             return self
#         return list(iter(self))
#
#     def __iter__(self) -> Generator[Groups, None, None]:
#         from validateosm.compare.compare import Compare
#         self._instance._instance: Compare
#         sources = self._instance._instance.sources
#         data = self._instance._instance.data
#
#         for name, source in sources.items():
#             structs = self.structs[source.__class__]
#             groups = (
#                 group
#                 for struct in structs
#                 for group in struct.func(source)
#                 if len(group) > 1
#             )
#             G = networkx.Graph()
#             for group in groups:
#                 G.add_nodes_from(group)
#                 G.add_edges_from(zip(group[:-1], group[1:]))
#             footprint = self._instance.footprint(data.xs(name))
#             G.add_nodes_from(footprint)
#             G.add_edges_from(footprint)
#             groups = connected_components(G)
#             ungrouped: list[int] = [
#                 group[0]
#                 for group in groups
#                 if len(group) == 1
#             ]
#             grouped: list[Collection[int]] = [
#                 group
#                 for group in groups
#                 if len(group) > 1
#             ]
#             yield Groups(
#                 _data=data,
#                 _grouped=grouped,
#                 _ungrouped=ungrouped
#             )
#

class DescriptorAggregate:

    def __get__(self, instance, owner) -> Union[GeoDataFrame, 'DescriptorAggregate']:
        from validateosm.compare.compare import Compare
        self._instance: Compare = instance
        self._owner: Type[Compare] = owner
        self._data = self._instance.data
        aggregate = self._aggregate
        if not self.path.exists():
            aggregate.to_feather(self.path)
        return aggregate

    @property
    def _structs(self) -> Generator[StructAggregate, None, None]:
        from validateosm.compare.compare import Compare
        self._instance: Compare
        sources = iter(self._instance.sources)
        first_source = next(sources)
        structs: dict[str, StructSingleAggregate] = getattr(first_source, '_aggregate')
        names: set[str] = set(structs.keys())
        for source in sources:
            matches = getattr(source, '_aggregate')
            if (dif := names.symmetric_difference(matches.keys())):
                raise ValueError(dif)
            for name, match in matches.items():
                struct = structs[name]
                for key, val in struct.__dict__.items():
                    if getattr(match, key) != val:
                        raise ValueError(
                            f"{source.__class__.__name__}.{name} != {first_source.__class__.__name__}.{name}"
                        )
        yield from (
            StructAggregate(**struct.__dict__)
            for struct in structs.values()
        )
    @functools.cached_property
    def _groups(self) -> list[Groups]:
        def groups() -> Generator[Groups, None, None]:
            data = self._instance.data
            duplicated = data.index.duplicated(False)
            ungrouped = data[~duplicated]
            # TODO: Group based on duplicated indices, instead of Series
            grouped = ungrouped[data.]


        return list(groups())


    # @functools.cached_property
    # def _groups(self) -> list[Groups]:
    #     def gen():
    #         nonlocal self
    #         structs = CacheStructs()
    #         sources = self._instance.sources
    #         data = self._instance.data
    #
    #         for name, source in sources.items():
    #             structs = structs[source.__class__]
    #             groups = (
    #                 group
    #                 for struct in structs
    #                 for group in struct.func(source)
    #                 if len(group) > 1
    #             )
    #             G = networkx.Graph()
    #             for group in groups:
    #                 G.add_nodes_from(group)
    #                 G.add_edges_from(zip(group[:-1], group[1:]))
    #             footprint = self._instance.footprint(data.xs(name))
    #             G.add_nodes_from(footprint)
    #             G.add_edges_from(footprint)
    #             groups = connected_components(G)
    #             ungrouped: list[int] = [
    #                 group[0]
    #                 for group in groups
    #                 if len(group) == 1
    #             ]
    #             grouped: list[Collection[int]] = [
    #                 group
    #                 for group in groups
    #                 if len(group) > 1
    #             ]
    #             yield Groups(
    #                 _data=data,
    #                 _grouped=grouped,
    #                 _ungrouped=ungrouped
    #             )
    #
    #     return list(gen())

    @functools.cached_property
    def _aggregate(self) -> gpd.GeoDataFrame:
        if self.path.exists() and not self._instance.redo:
            return gpd.read_feather(self.path)
        groups = self._groups
        structs: list[StructAggregate] = list(self._structs)

        indie: dict[str, Series] = {
            struct.name: struct.decorated_func(groups)
            for struct in structs
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
        depend: set[StructAggregate] = {
            struct
            for struct in structs
            if struct.dependent
        }
        while depend:
            viable: set[StructAggregate] = {
                struct for struct in depend if not struct.dependent.difference(aggregate.columns)
            }
            if not viable:
                raise RuntimeError(f"Cannot resolve with cross-dependencies: {depend}")
            for struct in viable:
                series = struct.decorated_func(groups)
                aggregate[struct.name] = series.repeat(rows) if len(series) == 1 else series
            depend = {struct for struct in depend if struct not in viable}

        aggregate = self._instance._identify(aggregate)
        return aggregate

    def __delete__(self, instance):
        del self._aggregate

    @property
    def path(self) -> Path:
        from validateosm.compare.compare import Compare
        self._instance: Compare
        return self._instance.directory / (self.__class__.__name__ + '.feather')
