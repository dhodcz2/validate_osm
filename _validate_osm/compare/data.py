import dataclasses
import logging
import os
from pathlib import Path
from typing import Union, Type, Any, Iterable
from weakref import WeakKeyDictionary

import geopandas as gpd
import numpy as np
import pandas as pd
from geopandas import GeoDataFrame

from validate_osm.logger import logged_subprocess, logger
from validate_osm.source.source import Source
from validate_osm.util import concat

if False | False:
    from validate_osm.compare import Compare


@dataclasses.dataclass
class StructData:
    name: str
    dtype: str
    abstract: bool
    crs: Any
    dependent: set[str]

    def __hash__(self):
        return hash(self.name)

    def __eq__(self, other):
        return self.name == other

    def __repr__(self):
        return self.name


class DescriptorData:
    cache: WeakKeyDictionary[object, GeoDataFrame] = WeakKeyDictionary()

    def __init__(self):
        self.compare = None
        self.owner = None


    def __get__(self, instance: 'Compare', owner: Type['Compare']) -> Union[GeoDataFrame, 'DescriptorData']:
        if instance in self.cache:
            return self.cache[instance]
        self.compare = instance
        self.owner = owner
        if instance is None:
            return self

        path = self.path
        if not self.redo and path.exists():
            with logged_subprocess(f'reading {instance.name}.data from {path}', timed=False):
                self.__set__(instance, gpd.read_feather(path))
        else:
            instance.preprocess()
            with logged_subprocess(f'extracting {instance.name}.data'):
                self.__set__(instance, self.extract())
            with logged_subprocess(f'transforming {instance.name}.data'):
                # TODO: Wtf
                self.compare = instance
                self.owner = owner
                self.__set__(instance, self.transform())
            with logged_subprocess(f'loading {instance.name}.data', logging.DEBUG, False):
                self.load()

        return self.__get__(instance, owner)

        # path = self.path
        # if path.exists() and not self.redo:
        #     with logged_subprocess(f'reading {instance.name}.data from {path}', timed=False):
        #         self.__set__(instance, gpd.read_feather(path))
        # else:
        #     instance.preprocess()
        #     with logged_subprocess(f'building {instance.name}.data'):
        #         self.__set__(instance, self.extract())
        #         with logged_subprocess('getting footprints'):
        #             _ = instance.footprints
        #         with logged_subprocess('applying footprints'):
        #             self.__set__(instance, instance.footprint(self.__get__(instance, owner)))
        #         self.compare = instance  # TODO: Still necessary?
        #     self.__get__(instance, owner).to_feather(self.path)
        #
        # return self.__get__(instance, owner)

    def extract(self) -> GeoDataFrame:
        data = concat(self)
        inval = data['geometry'].isna() | data['centroid'].isna()
        if any(inval):
            msg = ', '.join(str(tuple) for tuple in data[inval].index)
            logger.warning(f'invalid warning: {msg}')
            data = data[~inval]

        if self.compare is None:
            raise ValueError

        return data

    def transform(self):
        compare = self.compare
        owner = self.owner
        with logged_subprocess('getting footprints'):
            _ = compare.footprints
        with logged_subprocess('applying footprints'):
            data = compare.footprint(self.__get__(compare, owner))
        # if self.compare is None:
        #     raise ValueError
        return data

    def load(self):
        if not self.compare:
            raise ValueError(self.compare)
        if self.compare.serialize:
            data = self.cache[self.compare]
            path = self.path
            if not path.parent.exists():
                os.makedirs(path.parent)
            with logged_subprocess(f'serializing {self.compare.name}.data'):
                data.to_feather(path)

    def __hash__(self):
        return hash(self.compare)

    def __eq__(self, other):
        return self.compare == other

    def __iter__(self):
        sources: Iterable[Source] = self.compare.sources.values()
        for source in sources:
            for data in source:
                data['name'] = source.name
                if 'id' not in data:
                    data['id'] = np.nan
                if 'group' not in data:
                    data['group'] = np.nan
                data = data.set_index(['name', 'id', 'group'], drop=True)
                yield data

    def __delete__(self, instance):
        if instance in self.cache:
            del self.cache[instance]

    def __set__(self, instance, gdf: GeoDataFrame):
        gdf['iloc'] = pd.Series(range(len(gdf)), dtype='int32', index=gdf.index)
        self.cache[instance] = gdf

    def __repr__(self):
        return f'{self.compare}.data'

    @property
    def path(self) -> Path:
        return self.compare.directory / 'data.feather'

    def delete(self):
        os.remove(self.path)

    @property
    def redo(self):
        for string in ('data', *self.compare.sources.keys()):
            if string in self.compare.redo:
                return True
        return False
