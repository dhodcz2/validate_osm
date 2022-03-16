import functools
import itertools
import os
from pathlib import Path
from typing import Iterator
from typing import Union, Type

import geopandas as gpd
import pandas as pd
from geopandas import GeoDataFrame

from validateosm.source.aggregate import AggregateFactory
from validateosm.source.groups import Groups
from validateosm.source.source import Source
from weakref import WeakKeyDictionary


class DescriptorAggregate:
    cache: WeakKeyDictionary[object, GeoDataFrame] = WeakKeyDictionary()

    def __get__(self, instance, owner) -> Union[GeoDataFrame, 'DescriptorAggregate']:
        self._instance = instance
        self._owner = owner
        if instance is None:
            return self

        agg = self.cache.setdefault(instance, self._aggregate())
        path = self.path
        if not path.exists() or self._instance.ignore_file:
            if not path.parent.exists():
                os.makedirs(path.parent)
            agg.to_feather(path)
        return agg

    def _groups(self) -> Groups:
        data = self._instance.data
        iloc = pd.Series(range(len(data)), index=data.index)
        names = iloc.index.unique('name')
        index = self._instance.footprint.footprints.index.name

        def groups():
            # For each unique name, group all matches of UBID, relation, or way
            for name in names:
                iloc_frag = iloc.xs(name, level='name')
                yield from (
                    iloc_frag.iloc[group].values
                    for group in iloc_frag.groupby(index).indices.values()
                    if len(group) > 1
                )

        grouped = list(groups())
        ungrouped = set(iloc.values).difference(itertools.chain.from_iterable(grouped))
        return Groups(data.copy(), grouped, ungrouped)

    def _aggregate(self) -> gpd.GeoDataFrame:
        if self.path.exists() and not self._instance.ignore_file:
            return gpd.read_feather(self.path)

        groups = self._groups()
        sources = self._instance.sources.values()
        factories: Iterator[tuple[AggregateFactory, Source]] = zip(
            (source.aggregate_factory for source in sources),
            sources
        )
        factory, source = next(factories)
        for other_factory, other_source in factories:
            if other_factory.__class__ is not factory.__class__:
                raise ValueError(f"{source.__class__.__name__}.factory!={other_source.__class__.name}.factory")
        result = factory(groups)
        result['iloc'] = range(len(result))
        return result

    def __delete__(self, instance):
        if instance in self.cache:
            del self.cache[instance]

    @property
    def path(self) -> Path:
        from validateosm.compare.compare import Compare
        self._instance: Compare
        # return self._instance.directory / (self.__class__.__name__ + '.feather')
        return (
                self._instance.directory /
                self.__class__.__name__ /
                f'{str(self._instance.bbox)}.feather'
        )
