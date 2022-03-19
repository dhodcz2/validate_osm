import functools
import itertools
from typing import Iterator, Type, Callable
from weakref import WeakKeyDictionary

import geopandas as gpd
import numpy as np
import pandas as pd
import psutil
from OSMPythonTools.overpass import OverpassResult, Overpass, Element

from validate_osm.source.resource import Resource
from validate_osm.source.source import Source


class FragmentBBox:

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


class DescriptorWays:
    type = 'way'
    overpass: Overpass = Overpass()
    ESTIMATE_COST_PER_ENTRY_B = 11000

    def __get__(self, instance: 'DynamicOverpassResource', owner: Type['DynamicOverpassResource']):
        self.source = instance.source
        self.resource = instance
        return self

    def __iter__(self) -> Iterator[OverpassResult]:
        from validate_osm.source.source_osm import SourceOSM
        self.source: SourceOSM
        fragments = FragmentBBox(self.source.bbox.data.ellipsoidal.bounds)
        while fragments:
            peak = fragments.peak()
            query = self.source.query(peak, type=self.type, appendix='out count;')
            estimate = self.overpass.query(query, timeout=120).countElements() * self.ESTIMATE_COST_PER_ENTRY_B
            if estimate > psutil.virtual_memory().free:
                fragments.split()
            else:
                query = self.source.query(fragments.pop(), type=self.type)
                result: OverpassResult = self.overpass.query(query, timeout=300)
                yield result


class DescriptorRelations(DescriptorWays):
    type = 'relation'

    __get__: 'DescriptorRelations'

    def __init__(self, *args, **kwargs):
        super(DescriptorWays, self).__init__(*args, **kwargs)
        self.groups = []

    # TODO: How can we differentiate between groups that are incomplete because they are cut off by the bbox
    #   versus groups that have members that were disqualified by the query?

    def __iter__(self) -> Iterator[OverpassResult]:
        self.groups = []
        for result in super(DescriptorRelations, self).__iter__():
            self.groups.extend(
                [
                    relation.typeIdShort(),
                    *(member.typeIdShort() for member in relation.members())
                ]
                for relation in result.elements()
            )
            yield result


class DecoratorEnumerative:
    f"""
    Any SourceOSM.data function that includes data that may be seperated by semicolons should be wrapped with 
        DecoratorEnumerative so that the process of extracting data may separate values, cast them, and include them
        into the resulting GeoDataFrame.
    """
    cache: WeakKeyDictionary[object, dict[str, dict[str, list[object]]]] = WeakKeyDictionary()
    # TODO: This must be column: id so that we may generate appendix column-wise
    """ source: { id: { column: list } } """

    def __init__(self, cast: type):
        self.cast = cast

    def __call__(self, func: Callable):
        # TODO: How do I use typeid instead of integer index as the key
        from validate_osm.source.source_osm import SourceOSM
        @functools.wraps(func)
        def wrapper(source: SourceOSM):
            cache = self.cache.setdefault(source, {})
            ids = source.id
            cast = self.cast
            name = func.__name__
            result: Iterator[str] = func(source)
            for id, val in zip(ids, result):
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
                    vals = [cast(v) for v in val[1:]]
                except ValueError:
                    # TODO: Log
                    yield np.nan
                    continue
                cache.setdefault(id, {})[name] = vals
                yield v

        return wrapper


class DescriptorEnumerative:
    def __get__(self, instance: 'DynamicOverpassResource', owner: Type['DynamicOverpassResource']):
        self.instance = instance
        self.owner = owner
        self.source = self.instance.source
        return self.appendix()

    def appendix(self) -> gpd.GeoDataFrame:
        cache = DecoratorEnumerative.cache.setdefault(self.source, {})
        if not len(cache):
            self.source.logger.warning(
                f"{self.source.__class__.__name__}.resource has no enumerative entries; this is an unlikely outcome."
            )
        lens = {
            id: max(len(list) for list in column.values())
            for id, column in cache.items()
        }
        index = pd.Index((
            id
            for id, length in lens.items()
            for _ in range(length)
        ), name='id')
        data = self.source.data

        def gen(name, column) -> Iterator[pd.Series]:
            for id, length in lens.items():
                if name in cache[id]:
                    values = cache[id][name]
                    yield from values
                    if (leftover := length - len(values)) != 0:
                        yield from itertools.repeat(np.nan, leftover)
                else:
                    value = column[id]
                    for _ in range(length):
                        yield value

        return gpd.GeoDataFrame({
            name: (
                gpd.GeoSeries(gen(name, column), index=index, crs=column.crs)
                if isinstance(column, gpd.GeoSeries)
                else pd.Series(gen(name, column), index=index, dtype=column.dtype)
            )
            for name, column in data.iteritems()
        })


class DynamicOverpassResource(Resource):
    ways = DescriptorWays()
    relations = DescriptorRelations()
    enumerative = DescriptorEnumerative()
    name = 'osm'
    link = 'https://www.openstreetmap.org'

    def __get__(self, instance: Source, owner: Type[Source]):
        self.source = instance
        self.owner = owner
        return self

    def __iter__(self) -> Iterator[Element]:
        # Because Overpass caches identical query responses, we can load/unload every result without network load.
        #   Perhaps row-wise construction is more efficient for building the DataFrame because reloading is not
        #   necessary? Regardless, this is simpler to code and can be optimized later.
        for result in self.ways:
            yield from result.elements()
        for result in self.relations:
            yield from result.elements()

    def __delete__(self, instance):
        pass
