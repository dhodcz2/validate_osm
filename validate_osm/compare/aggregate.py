from functools import lru_cache

import logging
import os
from pathlib import Path
from typing import Iterator, Optional, Type
from typing import Union
from weakref import WeakKeyDictionary

import geopandas as gpd
import pandas as pd
from geopandas import GeoDataFrame

from validate_osm.source.aggregate import FactoryAggregate
from validate_osm.source.resource_ import StructFile, StructFiles
from validate_osm.source.source import Source
from validate_osm.logger import logger, logged_subprocess

if False:
    from validate_osm import Compare


class DescriptorAggregate:
    cache: WeakKeyDictionary[object, GeoDataFrame] = WeakKeyDictionary()

    @property
    def redo(self) -> bool:
        for string in ('aggregate', 'data', *self.compare.sources.keys()):
            if string in self.compare.redo:
                return True
        return False

    def __get__(self, instance: 'Compare', owner: Type['Compare']) -> Union[GeoDataFrame, 'DescriptorAggregate']:
        if instance in self.cache:
            return self.cache[instance]

        self.compare = instance
        self.owner = owner
        if instance is None:
            return self

        path = self.path
        if self.redo or not path.exists():
            instance.preprocess()
            self.__set__(instance, self.transform())
            self.load()
        else:
            with logged_subprocess(f'reading {instance.name}.aggregate from {path}', timed=False):
                self.__set__(instance, gpd.read_feather(self.path))
        return self.__get__(instance, owner)

    @property
    @lru_cache()
    def factory(self) -> FactoryAggregate:
        sources = self.compare.sources.values()
        factories: Iterator[tuple[Type[FactoryAggregate], Source]] = zip(
            (source.aggregate_factory for source in sources),
            sources
        )
        factory, source = next(factories)
        for other_factory, other_source in factories:
            if other_factory.__class__ is not factory.__class__:
                raise ValueError(f"{source.__class__.__name__}.factory!={other_source.__class__.name}.factory")
        logger.debug(f'using {factory.__name__}')
        # return factory(self.instance)
        return factory()

    def transform(self) -> GeoDataFrame:
        with logged_subprocess(f'building {self.compare.name}.aggregate'):
            return self.factory(self.compare)

    def load(self):
        if not self.compare:
            raise ValueError(self.compare)
        if self.compare.serialize:
            agg = self.cache[self.compare]
            path = self.path
            if not path.parent.exists():
                os.makedirs(path.parent)
            with logged_subprocess(f'serializing {self.compare.name}.aggregate'):
                agg.to_feather(path)

    def __delete__(self, instance):
        if instance in self.cache:
            del self.cache[instance]

    def __set__(self, instance, gdf: GeoDataFrame):
        gdf['iloc'] = pd.Series(range(len(gdf)), dtype='int32', index=gdf.index)
        self.cache[instance] = gdf

    @property
    def path(self) -> Path:
        return self.compare.directory / 'aggregate.feather'
