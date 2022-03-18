from networkx import connected_components, Graph
import abc
import functools
import itertools
from datetime import datetime
from typing import Iterable, Type, Optional, Hashable
from typing import Iterator, Callable
from typing import ValuesView, Union, Collection, Generator
from weakref import WeakKeyDictionary

import geopandas as gpd
import networkx
import numpy as np
import pandas as pd
import psutil
import shapely
from OSMPythonTools.overpass import OverpassResult, Overpass
from annoy import AnnoyIndex
from geopandas import GeoDataFrame, GeoSeries
from pandas import Series
from shapely.geometry import Polygon

from validate_osm.source.data import CacheData
from validate_osm.source.groups import (
    DecoratorGroup
)
from validate_osm.source.source import DescriptorData
from validate_osm.source.source import Source
from validate_osm.source.overpass import DynamicOverpassResource


class DecoratorEnumerative:
    f"""
    Any SourceOSM.data function that includes data that may be seperated by semicolons should be wrapped with 
        DecoratorEnumerative so that the process of extracting data may separate values, cast them, and include them
        into the resulting GeoDataFrame.
    """

    def __init__(self, cast: type):
        self.cast = cast
        self.i = 0

    def __call__(self, func: Callable):
        @functools.wraps(func)
        def wrapper(source: SourceOSM):
            cast = self.cast
            name = func.__name__
            result: Iterator[str] = func(source)
            for i, val in enumerate(result, start=self.i):
                # TODO: perhaps == '' is a worthwhile optimization
                if not val:
                    yield np.nan
                    continue
                if ';' not in val:
                    try:
                        yield cast(val)
                    except ValueError:
                        yield np.nan
                    continue
                val = val.split(';')
                try:
                    v = cast(val)
                    val = [cast(v) for v in val[1:]]
                except ValueError:
                    # TODO: Log
                    yield np.nan
                    continue
                source.enumerative[i, name] = val
                yield v
            self.i = i

        return wrapper


class DescriptorEnumerative:
    """
    This attribute extends the functionality of SourceOSM.data so that OpenStreetMaps tags with semicolon-separated-
        values may be accounted for.
    """
    _cache: WeakKeyDictionary[object, dict[str, dict[int, list[object]]]] = WeakKeyDictionary()

    def __get__(self, instance, owner):
        # Enumerative must be a descriptor for SourceOSM as the decorator only has access to the
        #   instance in its methods. Direct class access is an antipattern
        if instance is not getattr(self, '_instance', None):
            try:
                del self._lens
            except AttributeError:
                pass
        self._instance: SourceOSM = instance
        self._owner: Type[SourceOSM] = owner
        self._columns_rows_lists = self._cache.setdefault(self._instance.resource, {})
        return self

    def __setitem__(self, key, value):
        row, column = key
        self._columns_rows_lists.setdefault(column, {})[row] = value

    @functools.cached_property
    def _lens(self) -> dict[int, int]:
        columns = self._columns_rows_lists.values()
        rows: set[int] = {
            row
            for col in columns
            for row in col.keys()
        }
        return {
            row: max(
                len(col[row])
                for col in columns
                if row in col
            )
            for row in rows
        }

    def _extend(self, column: Series) -> Union[Series, GeoSeries]:
        # Because this is extending Source.data, we are not worried about messing with the index, which will be
        #   established postmortem.
        def gen() -> Iterator[object]:
            row_items = self._columns_rows_lists[str(column.name)]
            for row, length in self._lens:
                if row not in row_items:
                    val = column[row]
                    for _ in range(length):
                        yield val
                    continue
                items = iter(row_items[row])
                for _ in range(length):
                    try:
                        yield next(items)
                    except StopIteration:
                        yield np.nan

        data = itertools.chain(iter(column), gen())
        if isinstance(column, GeoSeries):
            return GeoSeries(data, name=column.name, crs=column.crs)
        else:
            return Series(data, name=column.name, dtype=column.dtype)

    def append(self, data: GeoDataFrame) -> GeoDataFrame:
        return GeoDataFrame({
            col: self._extend(data[col])
            for col in data
        })


#
# class DescriptorWays:
#     """
#         Handles the extraction of Ways from the OverpassAPI, taking available memory into consideration when querying
#             with a divide-and-conquer approach.
#     """
#     ESTIMATE_COST_PER_ENTRY_B = 11000
#     element: str = 'way'
#     _containers: WeakKeyDictionary[object, set[int]] = WeakKeyDictionary()
#     _contained: WeakKeyDictionary[object, set[int]] = WeakKeyDictionary()
#     _overpass = Overpass()
#
#     class _FragmentBbox:
#
#         def __init__(self, bbox: tuple[int]):
#             self.stack: list[tuple[int]] = [bbox]
#
#         def peak(self) -> tuple[int]:
#             return self.stack[-1]
#
#         def pop(self) -> tuple[int]:
#             return self.stack.pop()
#
#         def split(self):
#             pop = self.stack.pop()
#             half = (pop[2] + pop[0]) / 2
#             bottom = (pop[0], pop[1], half, pop[3])
#             top = (half, pop[1], pop[2], pop[3])
#             self.stack.extend((bottom, top))
#
#         def __bool__(self):
#             return bool(self.stack)
#
#     def __get__(self, instance, owner):
#         # DescriptorWays must be a descriptor for SourceOSM so that SourceOSM may define self.raw within
#         #   SourceOSM.__init__
#         self._instance: SourceOSM = instance
#         self._owner: Type[SourceOSM] = owner
#         self.containers = self._containers.setdefault(instance, set())
#         self.contained = self._contained.setdefault(instance, set())
#         return self
#
#     def __iter__(self) -> Iterator[OverpassResult]:
#         fragments = self._FragmentBbox(self._owner.bbox.data.ellipsoidal)
#         while fragments:
#             peak = fragments.peak()
#             query = self._owner.query(peak, type=self.element, appendix='out count;')
#             estimate = self._overpass.query(query).countElements() * self.ESTIMATE_COST_PER_ENTRY_B
#             if estimate > psutil.virtual_memory().free:
#                 fragments.split()
#             else:
#                 query = self._owner.query(fragments.pop(), type=self.element)
#                 opr: OverpassResult = self._overpass.query(query)
#                 yield opr
#                 self.containers.update(self._instance.containers())
#                 self.contained.update(
#                     id
#                     for element in opr.elements()
#                     if (id := element.id()) not in self.containers
#                 )
#
#
# class DescriptorRelations(DescriptorWays):
#     """
#         Handles the extraction of Relations from the OverpassAPI, taking available memory into consideration when
#             querying with a divide-and-conquer approach. Relations take special consideration because the set of ways
#             implied by their one-to-many relations may be incomplete. Incompleteness may be due to:
#             - the way did not qualify in the ways (irrelevant tag)
#             - the relation is on the border of the bounding box
#     """
#     # TODO: Would it be beneficial to independently query for ways that do not qualify for the original query however
#     #   are implicated by the relation?
#     element = 'relation'
#     _cache_dependent_relations: WeakKeyDictionary[object, dict[int, set[int]]] = WeakKeyDictionary()
#     _cache_dependent_ways: WeakKeyDictionary[object, dict[int, set[int]]] = WeakKeyDictionary()
#     dependent_relations: dict[int, set[int]]
#     dependent_ways: dict[int, set[int]]
#
#     def __get__(self, instance, owner) -> 'DescriptorRelations':
#         get = super(DescriptorRelations, self).__get__(instance, owner)
#         self.dependent_relations = self._cache_dependent_relations.setdefault(self._instance, {})
#         self.dependent_ways = self._cache_dependent_ways.setdefault(self._instance, {})
#         return get
#
#     def __iter__(self) -> Iterator[OverpassResult]:
#         for opr in super(DescriptorRelations, self).__iter__():
#             self.dependent_relations.update({
#                 relation.id(): {
#                     ele.id()
#                     for ele in relation.members()
#                     if ele.element() == 'relation'
#                 }
#                 for relation in opr.relations()
#             })
#             self.dependent_ways.update({
#                 relation.id(): {
#                     ele.id()
#                     for ele in relation.members()
#                     if ele.element() == 'way'
#                 }
#                 for relation in opr.relations()
#             })
#             yield opr
#
#
class SourceOSM(Source, abc.ABC):
    resource = DynamicOverpassResource()
    name = 'osm'
    link = 'https://www.openstreetmap.org'

    def group(self) -> GeoDataFrame:
        data: GeoDataFrame = self.data
        data['id'] = self.id
        self.data = data = data.set_index('id')
        # Set ID as an index

        # append enumerative entries
        data = data.append(self.resource.enumerative)

        # Group together according to relations
        G = networkx.Graph()
        for group in self.resource.relations.groups:
            G.add_nodes_from(group)
            G.add_edges_from(zip(group[:-1], group[1:]))
        index = set(data.index)
        groups = connected_components(G)
        incomplete: Iterable[Hashable] = [
            value
            for group in groups
            if not all(id in index for id in group)
            for value in groups
        ]

        # Drop those who are part of incomplete groups
        data = data.drop(incomplete)
        complete: list[set] = [
            group for group in groups
            if all(id in index for id in group)
        ]
        data['group'] = pd.Series(
            data=(
                i
                for i, group in enumerate(complete)
                for member in group
            ), index=itertools.chain.from_iterable(complete)
        )
        data: gpd.GeoDataFrame = data.set_index('group', append=True)
        # Append enumerative groups
        return data

    @classmethod
    @abc.abstractmethod
    def query(cls, bbox: tuple[int], type='way', appendix: str = 'out meta geom;'):
        ...

    @functools.cached_property
    def id(self) -> Iterator[str]:
        return [
            element.typeIdShort()
            for element in self.resource
        ]

    def timestamp(self) -> Iterable[datetime]:
        generator: Iterator[datetime] = (
            # datetime.strptime(element.timestamp(), '%Y-%m-%dT%H:%M:%SZ')
            element.timestamp()
            for element in self.resource
        )
        return Series(generator, dtype='datetime64[ns]')

    def geometry(self) -> GeoSeries:
        mapping = {
            'Polygon': lambda coordinates: Polygon(coordinates[0]),
            'LineString': lambda coordinates: None,
            'MultiPolygon': lambda coordinates: shapely.geometry.MultiPolygon((
                Polygon(polygon[0])
                for polygon in coordinates
            ))
        }

        def generator():
            for element in self.resource:
                if not (element.type() == 'way' or element.tag('type') == 'multipolygon'):
                    yield None
                    continue
                try:
                    yield mapping[element.geometry()['type']](element.geometry()['coordinates'])
                    continue
                except KeyError as e:
                    raise NotImplementedError(element.geometry()['type']) from e
                except Exception:  # Exception is typically a no-go but .geometry() literally raises this
                    yield None

        gs = GeoSeries(generator(), crs=4326)
        return gs
