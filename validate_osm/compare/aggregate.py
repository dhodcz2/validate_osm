import os
from pathlib import Path
from typing import Iterator, Optional, Type
from typing import Union
from weakref import WeakKeyDictionary

import geopandas as gpd
from geopandas import GeoDataFrame

from validate_osm.source.aggregate import AggregateFactory
from validate_osm.source.resource import File
from validate_osm.source.source import Source
from validate_osm.util.scripts import logged_subprocess


class DescriptorAggregate:
    cache: WeakKeyDictionary[object, GeoDataFrame] = WeakKeyDictionary()

    def __get__(self, instance: Optional[object], owner: type) -> Union[GeoDataFrame, 'DescriptorAggregate']:
        from validate_osm.compare.compare import Compare
        instance: Optional[Compare]
        owner: Type[Compare]
        self._instance = instance
        self._owner = owner
        if instance is None:
            return self
        if instance in self.cache:
            return self.cache[instance]
        path = self.path
        if 'aggregate' in instance.redo or not path.exists():
            with logged_subprocess(instance.logger, f'building {instance.name}.aggregate'), self as aggregate:
                return aggregate
        else:
            with logged_subprocess(
                    instance.logger, f'reading {instance.name}.aggregate from {path} ({File.size(path)})', timed=False
            ):
                agg = self.cache[instance] = gpd.read_feather(self.path)
                return agg

    # TODO: Perhaps clean up this process a bit; it seems like it can be simplified with regards to dunder methods
    @property
    def factory(self) -> AggregateFactory:
        sources = self._instance.sources.values()
        factories: Iterator[tuple[Type[AggregateFactory], Source]] = zip(
            (source.aggregate_factory for source in sources),
            sources
        )
        factory, source = next(factories)
        for other_factory, other_source in factories:
            if other_factory.__class__ is not factory.__class__:
                raise ValueError(f"{source.__class__.__name__}.factory!={other_source.__class__.name}.factory")
        self._instance.logger.debug(f'using {factory.__name__}')
        return factory(self._instance)

    def __enter__(self) -> GeoDataFrame:
        with self.factory as aggregate:
            self.cache[self._instance] = aggregate
        return aggregate

    def __exit__(self, exc_type, exc_val, exc_tb):
        from validate_osm.compare.compare import Compare
        self._instance: Compare
        agg = self.cache[self._instance]
        if self._instance.serialize:
            path = self.path
            if not path.parent.exists():
                os.makedirs(path.parent)
            with logged_subprocess(self._instance.logger, f'serializing {self._instance.name}.aggregate to {path}'):
                agg.to_feather(path)

    def __delete__(self, instance):
        if instance in self.cache:
            del self.cache[instance]

    def __set__(self, instance, value):
        self.cache[instance] = value

    @property
    def path(self) -> Path:
        from validate_osm.compare.compare import Compare
        self._instance: Compare
        return self._instance.directory / str(self._instance.bbox) / 'aggregate.feather'
