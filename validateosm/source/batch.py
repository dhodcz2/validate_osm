from typing import Union
import geopandas as gpd
from weakref import WeakKeyDictionary
import inspect
from pathlib import Path

from geopandas import GeoDataFrame
from collections import UserDict
from validateosm.source.pipe import DescriptorPipeSerialize


class CacheBatch(UserDict):
    def __init__(self):
        super(CacheBatch, self).__init__()
        self.data: WeakKeyDictionary[object, GeoDataFrame] = WeakKeyDictionary()

    def __missing__(self, key):
        from validateosm.source import Source
        key: Source
        batch: GeoDataFrame = key.aggregate.copy()
        identity = key.identity()
        if identity is not None:
            batch.update(identity)
        exclude = key.exclude()
        if exclude is not None:
            batch = batch.loc[~exclude]
        self.data[key] = batch
        return batch


class DescriptorBatch(DescriptorPipeSerialize):
    _cache: CacheBatch[object, GeoDataFrame] = CacheBatch()

    def __get__(self, instance, owner) -> Union['DescriptorBatch', GeoDataFrame]:
        delagg = instance is not None and instance not in self._cache and not self.path(owner).exists()
        ret = super(DescriptorBatch, self).__get__(instance, owner)
        if delagg:
            del instance.aggegregate
        return ret
