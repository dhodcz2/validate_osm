import dataclasses
import logging
import os
from pathlib import Path
from typing import Generator, Union, Type, Any
from typing import Optional
from weakref import WeakKeyDictionary

import geopandas as gpd
import numpy as np
import pandas as pd
from geopandas import GeoDataFrame

from validate_osm.source import File
from validate_osm.util import concat
from validate_osm.util.scripts import logged_subprocess


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


# footprints/
class DescriptorData:
    cache: WeakKeyDictionary[object, GeoDataFrame] = WeakKeyDictionary()

    def __init__(self):
        self._instance = None
        self._owner = None

    def __get__(self, instance: object, owner: Type) -> Union[GeoDataFrame, 'DescriptorData']:
        if instance in self.cache:
            # TODO: does .copy() slow down significantly?
            return self.cache[instance]

        self._owner = owner
        if instance is None:
            return self

        from validate_osm.compare.compare import Compare
        owner: Optional[Type[Compare]]
        instance: Compare
        self._instance = instance
        path = self.path
        # TODO: Perhaps if there is a file with a bbox that contains instance.bbox, load
        #   that bbox, .within it,
        if path.exists() and 'data' not in instance.redo:
            with logged_subprocess(
                    instance.logger,
                    f'reading {instance.name}.data from {path} ({File.size(path)})',
                    timed=False
            ):
                self.__set__(instance, gpd.read_feather(path))
        else:
            with logged_subprocess(instance.logger, f'building {instance.name}.data'), self as data:
                with logged_subprocess(instance.logger, 'getting footprints'):
                    _ = self._instance.footprints
                with logged_subprocess(instance.logger, 'applying footprints'):
                    self.__set__(instance, instance.footprint(data))
        return self.__get__(instance, owner)

    # I like using enter/exit with descriptors because the __exit__ is the perfect place for serialization, logging,
    #   checking, etc. without creating spaghetti code that is incompatible with inheritance or added features.
    def __enter__(self) -> GeoDataFrame:
        # __enter__ is before the data is footprinted
        from validate_osm.compare.compare import Compare
        self._instance: Compare
        self._owner: Type[Compare]

        # We must concatenate DataFrames instead of going purely by Series because some columns
        #   may return a single-value, which must be repeated to the same length of other columns
        # To preserve SourceOSM.relation and SourceOSM.way indices, we must first
        def datas() -> Generator[gpd.GeoDataFrame, None, None]:
            for name, source in self._instance.sources.items():
                data: GeoDataFrame = source.data
                data = data.reset_index()
                data['name'] = name
                if 'id' not in data:
                    data['id'] = np.nan
                if 'group' not in data:
                    data['group'] = np.nan
                data = data.set_index(['name', 'id', 'group'], drop=True)
                yield data
                # TODO: For some reason this logger.debug was causing me errors?
                # sourcename = source.name
                # logger = self._instance.logger
                # logger.debug(f'deleting {sourcename}.resource')
                del source.data

        with logged_subprocess(self._instance.logger, f'concatenating {self._instance.name}.data', ):
            data = concat(datas())

        # TODO: Investigate how these invalid geometries came to be
        inval = data['geometry'].isna() | data['centroid'].isna()
        if any(inval):
            msg = ', '.join(str(tuple) for tuple in data[inval].index)
            self._instance.logger.warning(f'invalid geometry: {msg}')
            data = data[~inval]

        self.__set__(self._instance, data)
        return self.__get__(self._instance, 'data')

    def __exit__(self, exc_type, exc_val, exc_tb):
        from validate_osm.compare.compare import Compare
        self._instance: Compare
        data = self.cache[self._instance]
        if self._instance.serialize:
            path = self.path
            if not path.parent.exists():
                os.makedirs(path.parent)
            with logged_subprocess(
                    self._instance.logger,
                    f'serializing {self._owner.__name__}.data {path}',
                    timed=False
            ):
                data.to_feather(path)

    def __delete__(self, instance):
        if instance in self.cache:
            del self.cache[instance]

    def __set__(self, instance, gdf: GeoDataFrame):
        gdf['iloc'] = pd.Series(range(len(gdf)), dtype='int32', index=gdf.index)
        self.cache[instance] = gdf

    def __repr__(self):
        return f'{self._instance}.data'

    @property
    def path(self) -> Path:
        from validate_osm.compare.compare import Compare
        self._instance: Compare
        return self._instance.directory / str(self._instance.bbox) / 'data.feather'

    def delete(self):
        os.remove(self.path)
