import functools
import functools
import itertools
from pathlib import Path
from typing import Iterator
from typing import Union, Type

import geopandas as gpd
import pandas as pd
from geopandas import GeoDataFrame

from validateosm.source.aggregate import AggregateFactory
from validateosm.source.groups import Groups
from validateosm.source.source import Source


class DescriptorAggregate:

    def __get__(self, instance, owner) -> Union[GeoDataFrame, 'DescriptorAggregate']:
        from validateosm.compare.compare import Compare
        self._instance: Compare = instance
        self._owner: Type[Compare] = owner
        self._data = self._instance.data
        aggregate = self.aggregate
        return aggregate

    @property
    def groups(self) -> Groups:
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

    @functools.cached_property
    def aggregate(self) -> gpd.GeoDataFrame:
        if self.path.exists() and not self._instance.ignore_file:
            return gpd.read_feather(self.path)
        groups = self.groups

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

        result.to_feather(self.path)
        return result

    def __delete__(self, instance):
        del self.aggregate

    @property
    def path(self) -> Path:
        from validateosm.compare.compare import Compare
        self._instance: Compare
        return self._instance.directory / (self.__class__.__name__ + '.feather')
