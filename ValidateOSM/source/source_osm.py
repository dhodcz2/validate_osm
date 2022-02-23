import geopandas as gpd
import dataclasses
import math
from collections import UserList
from weakref import WeakKeyDictionary
from ValidateOSM.source.data import CacheData
from typing import ValuesView, Union, Collection, Generator

import psutil
from ValidateOSM.source.source import DescriptorData

import abc
import functools
import itertools
# from collections import Iterable, Generator
from typing import Iterable, Type
from datetime import datetime
from typing import Iterator, Callable

import numpy as np
import pandas as pd
import shapely
from OSMPythonTools.overpass import OverpassResult, Overpass
from annoy import AnnoyIndex
from geopandas import GeoDataFrame, GeoSeries
from pandas import Series
from shapely.geometry import Polygon

from ValidateOSM.source.source import Source
from ValidateOSM.source.groups import (
    DecoratorGroup
)


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


class DescriptorWays:
    """
        Handles the extraction of Ways from the OverpassAPI, taking available memory into consideration when querying
            with a divide-and-conquer approach.
    """
    ESTIMATE_COST_PER_ENTRY_B = 11000
    element: str = 'way'
    _containers: WeakKeyDictionary[object, set[int]] = WeakKeyDictionary()
    _contained: WeakKeyDictionary[object, set[int]] = WeakKeyDictionary()
    _overpass = Overpass()

    class _FragmentBbox:

        def __init__(self, bbox: tuple[int]):
            self.stack: list[tuple[int]] = [bbox]

        def peak(self) -> tuple[int]:
            return self.stack[-1]

        def pop(self) -> tuple[int]:
            return self.stack.pop()

        def split(self):
            pop = self.stack.pop()
            half = (pop[2] + pop[0]) / 2
            bottom = (pop[0], pop[1], half, pop[3])
            top = (half, pop[1], pop[2], pop[3])
            self.stack.extend((bottom, top))

        def __bool__(self):
            return bool(self.stack)

    def __get__(self, instance, owner):
        # DescriptorWays must be a descriptor for SourceOSM so that SourceOSM may define self.raw within
        #   SourceOSM.__init__
        self._instance: SourceOSM = instance
        self._owner: Type[SourceOSM] = owner
        self.containers = self._containers.setdefault(instance, set())
        self.contained = self._contained.setdefault(instance, set())
        return self

    def __iter__(self) -> Iterator[OverpassResult]:
        fragments = self._FragmentBbox(self._owner.bbox.data.ellipsoidal)
        while fragments:
            peak = fragments.peak()
            query = self._owner.query(peak, type=self.element, appendix='out count;')
            estimate = self._overpass.query(query).countElements() * self.ESTIMATE_COST_PER_ENTRY_B
            if estimate > psutil.virtual_memory().free:
                fragments.split()
            else:
                query = self._owner.query(fragments.pop(), type=self.element)
                opr: OverpassResult = self._overpass.query(query)
                yield opr
                self.containers.update(self._instance.containers())
                self.contained.update(
                    id
                    for element in opr.elements()
                    if (id := element.id()) not in self.containers
                )


class DescriptorRelations(DescriptorWays):
    """
        Handles the extraction of Relations from the OverpassAPI, taking available memory into consideration when
            querying with a divide-and-conquer approach. Relations take special consideration because the set of ways
            implied by their one-to-many relations may be incomplete. Incompleteness may be due to:
            - the way did not qualify in the ways (irrelevant tag)
            - the relation is on the border of the bounding box
    """
    # TODO: Would it be beneficial to independently query for ways that do not qualify for the original query however
    #   are implicated by the relation?
    element = 'relation'
    _cache_dependent_relations: WeakKeyDictionary[object, dict[int, set[int]]] = WeakKeyDictionary()
    _cache_dependent_ways: WeakKeyDictionary[object, dict[int, set[int]]] = WeakKeyDictionary()
    dependent_relations: dict[int, set[int]]
    dependent_ways: dict[int, set[int]]

    def __get__(self, instance, owner) -> 'DescriptorRelations':
        get = super(DescriptorRelations, self).__get__(instance, owner)
        self.dependent_relations = self._cache_dependent_relations.setdefault(self._instance, {})
        self.dependent_ways = self._cache_dependent_ways.setdefault(self._instance, {})
        return get

    def __iter__(self) -> Iterator[OverpassResult]:
        for opr in super(DescriptorRelations, self).__iter__():
            self.dependent_relations.update({
                relation.id(): {
                    ele.id()
                    for ele in relation.members()
                    if ele.element() == 'relation'
                }
                for relation in opr.relations()
            })
            self.dependent_ways.update({
                relation.id(): {
                    ele.id()
                    for ele in relation.members()
                    if ele.element() == 'way'
                }
                for relation in opr.relations()
            })
            yield opr


class CacheNeedlesData(CacheData):

    def __init__(self):
        super(CacheNeedlesData, self).__init__()
        self._instance: SourceOSM
        self._source: Type[SourceOSM]

    def __missing__(self, key: 'SourceOSM'):
        self.key = key
        return super(CacheNeedlesData, self).__missing__(key)

    def drop_incomplete_relations(self, data: GeoDataFrame) -> GeoDataFrame:
        key = self.key
        relations = key.relations.containers.union(key.relations.contained)
        ways = key.ways.containers.union(key.ways.contained)
        from_ways = {
            parent
            for parent, children in key.relations.dependent_ways.items()
            if not all(child in ways for child in children)
        }
        from_relations = (
            relation
            for parent, children in key.relations.dependent_relations.items()
            if not all(child in relations for child in children)
            for relation in (parent, *children)
        )
        incomplete_relations = itertools.chain(from_relations, from_ways)
        incomplete_ways = (
            child
            for parent, children in key.relations.dependent_ways.items()
            if parent in from_ways
            for child in children
        )
        data = data.drop(incomplete_ways, level='way', errors='ignore')
        data = data.drop(incomplete_relations, level='relation', errors='ignore')
        return data

    def resolve(self) -> GeoDataFrame:
        data = super(CacheNeedlesData, self).resolve()
        data = self.key.enumerative.append(data)
        data: GeoDataFrame = data.set_index(['way', 'relation'], append=True)
        data = self.drop_incomplete_relations(data)
        return data


class DescriptorNeedlesData(DescriptorData):
    _cache_data: WeakKeyDictionary[object, GeoDataFrame] = CacheNeedlesData()

class NonStaticOverpass():
    def __iter__(self) -> Generator[OverpassResult, None, None]:
        ...



# TODO: Instead of Source.ways and Source.relations, resource=NonStaticOverpass
class SourceOSM(Source, abc.ABC):
    data = DescriptorNeedlesData()
    enumerative = DescriptorEnumerative()
    ways = DescriptorWays()
    relations = DescriptorRelations()
    name = 'osm'
    link = 'https://www.openstreetmap.org'


    @functools.cached_property
    def resource(self) -> Union[OverpassResult, Iterator[OverpassResult]]:
        raise NotImplementedError
        # return itertools.chain(self.ways, self.relations)

    @classmethod
    @abc.abstractmethod
    def query(cls, bbox: tuple[int], type='way', appendix: str = 'out meta geom;'):
        ...

    @abc.abstractmethod
    def containers(self) -> Iterable[int]:
        """Return the element ID of elements that will be considered geometric containers"""

    def timestamp(self) -> Iterable[datetime]:
        generator: Iterator[datetime] = (
            datetime.strptime(element.timestamp(), '%Y-%m-%dT%H:%M:%SZ')
            for element in self.resource
        )
        return Series(generator, dtype='datetimens[64]')

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

    @DecoratorGroup(name='way')
    def _(self) -> ValuesView[Collection[int]]:
        return self.data.groupby('way', dropna=True).indices.values()

    @DecoratorGroup(name='relation')
    def _(self) -> ValuesView[Collection[int]]:
        return self.data.groupby('relation', dropna=True).indices.values()

    @DecoratorGroup(name='footprint')
    # TODO: self.footprint = None or self.footprint = Other
    def _(self) -> ValuesView[Collection[int]]:
        if self.footprint is not None and self.footprint is not self.__class__:
            # Problem: how do we handle inheritance?
            return super(SourceOSM, self).footprint()

        data: GeoDataFrame = self.data

        way = data.index.get_level_values('way')
        relations = data[way.isna()]
        relations: GeoDataFrame = relations[~relations.index.get_level_values('relation').duplicated()]
        ways = data[way.notna()]
        ways: GeoDataFrame = ways[~ways.index.get_level_values('way').duplicated()]
        data = relations.append(ways)

        data['geometry'] = data.geometry.to_crs(3857)
        data['area'] = data.area
        data['buffer'] = data.buffer(1)
        data['containment'] = pd.Series(np.nan, dtype='Int64', index=data.index)

        containers = data[data['container']]
        contained = data[~data['container']]
        containers.sort_values('area', ascending=False)
        containers['containment'] = range(len(containers))  # 0, 1, 2, 3...

        # TODO: We are going to first build containment using container absolute index, and then update the
        #   the containment column for all the rows.
        annoy = AnnoyIndex(2, 'euclidean')
        for i, centroid in enumerate(containers['centroid']):
            annoy.add_item(i, (centroid.x, centroid.y))
        annoy.build(10)

        for i, (g, r) in enumerate(containers.geometry, containers['relation']):
            for n in annoy.get_nns_by_item(i, 5):
                if n > i:
                    continue
                if (containers.loc[n, 'relation'] == r) is True:
                    continue
                if containers.loc[n, 'buffer'].contains(g):
                    containers.loc[i, 'containment'] = containers.loc[n, 'containment']
        data.loc[containers.index, 'containment'] = containers['containment']

        for i, g, r, a, c in zip(
                contained.index,
                contained.geometry,
                contained['relation'],
                contained['area'],
                contained['centroid']
        ):
            for n in annoy.get_nns_by_vector((c.x, c.y), 20):
                if (containers.loc[n, 'relation'] == r) is True:
                    continue
                if containers.loc[n, 'area'] < a:
                    continue
                if containers.loc[n, 'buffer'].contains(g):
                    contained.loc[i, 'containment'] = containers.loc[n, 'containment']

        # data.loc[contained.index, 'containment'] = contained['containment']

        return data.groupby('containment').indices.values()

    # TODO:
    @DecoratorGroup('footprint')
    def _(self) -> ValuesView[Collection[int]]:
        data: GeoDataFrame = self.data
        _way = data.index.get_level_values('way')
        relations = data[_way.isna()]
        relations: GeoDataFrame = relations[~relations.index.get_level_values('relation').duplicated()]
        ways = data[_way.notna()]
        ways: GeoDataFrame = ways[~ways.index.get_level_values('way').duplicated()]
        data = relations.append(ways)

        data['geometry'] = data.geometry.to_crs(3857)
        data['area'] = data.area

        if self.footprint is not None and self.footprint is not self.__class__:
            footprint = pd.Series(index=data.index)
            # Use external footprint
            try:
                external = self.footprint._footprint
                annoy = self.footprint._annoy
            except AttributeError:
                external: gpd.GeoDataFrame = self.footprint.aggregate[['geometry', 'centroid']]
                external['geometry'] = external.to_crs(3857)
                annoy = AnnoyIndex(2, 'euclidean')
                for i, centroid in enumerate(external['centroid']):
                    annoy.add_item(i, (centroid.x, centroid.y))
                annoy.build(10)
                self.footprint._annoy = annoy
                self.footprint._footprint = external

            for i, (c, a, g) in enumerate(data[['centroid', 'area', 'geometry']].values):
                for n in annoy.get_nns_by_vector((c.x, c.y), 5):
                    external = external.iloc[n]
                    if not external['geometry'].intersects(g):
                        continue
                    if external['geometry'].intersection(g).area / a < .5:
                        continue
                    footprint.iloc[i] = footprint[n]

            # Exclude anything that is not encapsulated by the external footprint
            footprint = footprint[footprint.duplicated(keep=False)]


        else:
            # Use internal footprint
            footprint = pd.Series(index=data.index)
            data['buffer'] = data.buffer(1)
            data['footprint'] = pd.Series(np.nan, dtype='Int64', index=data.index)
            containers = data[data['container']]
            contained = data[~data['container']]
            containers.sort_values('area', ascending=False)
            containers['footprint'] = range(len(containers))  # 0, 1, 2, 3...
            # TODO: We are going to first build footprint using container absolute index, and then update the
            #   the footprint column for all the rows.
            annoy = AnnoyIndex(2, 'euclidean')
            for i, centroid in enumerate(containers['centroid']):
                annoy.add_item(i, (centroid.x, centroid.y))
            annoy.build(10)

            for i, (g, r) in enumerate(containers.geometry, containers['relation']):
                for n in annoy.get_nns_by_item(i, 5):
                    if i < n:
                        continue
                    if (containers.loc[n, 'relation'] == r) is True:  # Same relation redundant
                        continue
                    if not containers.loc[n, 'buffer'].contains(g):
                        continue
                    containers.loc[i, 'footprint'] = containers.loc[n, 'footprint']
            footprint.update(containers['footprint'])

            for i, g, r, a, c in zip(
                    contained.index,
                    contained.geometry,
                    contained['relation'],
                    contained['area'],
                    contained['centroid']
            ):
                for n in annoy.get_nns_by_vector((c.x, c.y), 20):
                    if (containers.loc[n, 'relation'] == r) is True:
                        continue
                    if containers.loc[n, 'area'] < a:
                        continue
                    if containers.loc[n, 'buffer'].contains(g):
                        contained.loc[i, 'footprint'] = containers.loc[n, 'footprint']
            footprint.update(contained['footprint'])

        footprint = footprint[self.data.index]  # Retain original order because groupby.indices returns iloc
        return footprint.groupby(footprint, dropna=True).indices.values()
