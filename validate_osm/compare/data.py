import logging
from typing import Iterator
from validate_osm.source.source import Source
from validate_osm.source.footprint import CallableFootprint
from weakref import WeakKeyDictionary
import dataclasses
import os
import warnings
from pathlib import Path
from typing import Generator, Union, Type, Any

import geopandas as gpd
import numpy as np
from geopandas import GeoDataFrame

from validate_osm.util import concat


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
        self._instance = None
        self._owner = None

    def __get__(self, instance: object, owner: Type) -> Union[GeoDataFrame, 'DescriptorData']:
        self._instance = instance
        self._owner = owner
        if instance is None:
            return self
        if instance in self.cache:
            return self.cache[instance]
        path = self.path
        # TODO: Perhaps if there is a file with a bbox that contains instance.bbox, load
        #   that bbox, .within it,
        if path.exists() and not instance.ignore_file:
            logging.info(f'reading {owner.__name__}.data from {path}')
            data = self.cache[instance] = gpd.read_feather(path)
        else:
            logging.info(f'building {owner.__name__}.data')
            data = self.cache[instance] = self._data()
            if not path.parent.exists():
                os.makedirs(path.parent)
            logging.info(f'serializing {owner.__name__}.data {path}')
            data.to_feather(path)
        return data

    def _data(self) -> GeoDataFrame:
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
                logging.debug(f'{self.__class__.__name__}.data done; deleting {source.resource.__class__.__name__}')
                del source.data

        logging.debug(f'concatenating {self.__class__.__name__}.data')
        data = concat(datas())

        # TODO: Investigate how these invalid geometries came to be
        inval = data[data['geometry'].isna() | data['centroid'].isna()].index
        if len(inval):
            logging.warning(f'no geom: {inval}')
            data = data.drop(index=inval)


        # Instantiate Compare.footprint; use it to group Compare.data
        sources = self._instance.sources.values()
        footprints: Iterator[tuple[Type[CallableFootprint], Source]] = zip(
            (source.footprint for source in sources), sources
        )
        footprint, source = next(footprints)
        for other_footprint, other_source in footprints:
            if other_footprint is not footprint:
                raise ValueError(f"{source.__class__.__name__}.footprint!={other_source.__class__.name}.footprint")
        footprint = self._instance.footprint = footprint(data)

        data = footprint(data)
        data = data.sort_index(axis=0)
        data['iloc'] = range(len(data))
        data['geometry'] = data['geometry'].to_crs(3857)
        data['centroid'] = data['centroid'].to_crs(3857)
        return data

    def __delete__(self, instance):
        if instance in self.cache:
            del self.cache[instance]

    def __set__(self, instance, value):
        self.cache[instance] = value

    def __repr__(self):
        return f'{self._instance}.data'

    def __bool__(self):
        return self._instance in self.cache

    @property
    def path(self) -> Path:
        from validate_osm.compare.compare import Compare
        self._instance: Compare
        # return self._instance.directory / (self.__class__.__name__ + '.feather')
        return (
                self._instance.directory /
                self.__class__.__name__ /
                f'{str(self._instance.bbox)}.feather'
        )

    def delete(self):
        os.remove(self.path)
