import itertools
import logging
import os
from pathlib import Path
from typing import Iterator
from typing import Union
from weakref import WeakKeyDictionary

import geopandas as gpd
import pandas as pd
from geopandas import GeoDataFrame
from python_log_indenter import IndentedLoggerAdapter

from validate_osm.source.aggregate import AggregateFactory
from validate_osm.source.groups import Groups
from validate_osm.source.source import Source
from validate_osm.util.scripts import logged_subprocess

logger = logging.getLogger(__name__.partition('.')[0])
logger = IndentedLoggerAdapter(logger)


class DescriptorAggregate:
    cache: WeakKeyDictionary[object, GeoDataFrame] = WeakKeyDictionary()

    def __get__(self, instance, owner) -> Union[GeoDataFrame, 'DescriptorAggregate']:
        self._instance = instance
        self._owner = owner
        if instance is None:
            return self
        if instance in self.cache:
            return self.cache[instance]
        path = self.path
        if instance.ignore_file or not path.exists():
            with logged_subprocess(logger, f'building {owner}.data'):
                agg = self.cache[instance] = self._aggregate()

            if not path.parent.exists():
                os.makedirs(path.parent)
            with logged_subprocess(logger, f'serializing {owner}.data to {path}'):
                agg.to_feather(path)
        else:
            with logged_subprocess(logger, f'reading {owner}.aggregate from {path}'):
                agg = self.cache[instance] = gpd.read_feather(self.path)
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
        result['geometry'] = result['geometry'].to_crs(3857)
        result['centroid'] = result['centroid'].to_crs(3857)
        return result

    def __delete__(self, instance):
        if instance in self.cache:
            del self.cache[instance]

    @property
    def path(self) -> Path:
        from validate_osm.compare.compare import Compare
        self._instance: Compare
        return (
                self._instance.directory /
                self.__class__.__name__ /
                f'{str(self._instance.bbox)}.feather'
        )
