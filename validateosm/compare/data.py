import dataclasses
import os
import warnings
from pathlib import Path
from typing import Generator, Union, Type, Any

import geopandas as gpd
import numpy as np
from geopandas import GeoDataFrame

from validateosm.util import concat


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
    def __init__(self):
        self._instance = None
        self._owner = None
        self._data = None

    def __get__(self, instance, owner) -> Union[GeoDataFrame, 'DescriptorData']:
        self._instance = instance
        self._owner = owner
        if self._instance is None:
            return self
        elif self._data is not None:
            return self._data
        elif self.path.exists() and not instance.ignore_file:
            self._data = data = gpd.read_feather(self.path)
            warnings.warn(f"{repr(self._instance)}.data has been loaded from cache. To force a redo, assign "
                          f"{repr(self._instance)}=True")
            return data
        else:
            from validateosm.compare.compare import Compare
            self._instance: Compare = instance
            self._owner: Type[Compare] = owner

            # We must concatenate DataFrames instead of going purely by Series because some columns
            #   may return a single-value, which must be repeated to the same length of other columns
            # To preserve SourceOSM.relation and SourceOSM.way indices, we must first
            def datas() -> Generator[gpd.GeoDataFrame, None, None]:
                for name, source in self._instance.sources.items():
                    data: GeoDataFrame = source.data
                    data['name'] = name
                    if 'relation' not in data:
                        data['relation'] = np.nan
                    if 'way' not in data:
                        data['way'] = np.nan
                    yield data.set_index(['name', 'relation', 'way'])
                    del source.data

            # self._data must be set for footprint to not cause inifinite recursion
            try:
                data = self._data = concat(datas())
                data = self._data = self._instance.footprint(data)
            except Exception as e:
                # If something happens, we don't want a half-formed _data
                del self._data
                raise e

            data = self._data = data.sort_index(axis=0)

            if not self.path.parent.exists():
                os.makedirs(self.path.parent)
            data.to_feather(self.path)
            return data

    def __delete__(self, instance):
        del self._data

    def __repr__(self):
        return f'{self._instance}.data'

    def __bool__(self):
        return self._data is not None

    # TODO: Compare.data should not have the identifier directly applied to the data. Instead, the identifier
    #   is applied to the footprint that groups the data; data:footprint is many-to-one; aggregate:footprint is
    #   one-to-one.

    @property
    def path(self) -> Path:
        from validateosm.compare.compare import Compare
        self._instance: Compare
        return self._instance.directory / (self.__class__.__name__ + '.feather')

    def delete(self):
        os.remove(self.path)
