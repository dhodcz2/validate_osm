from typing import Iterator

from pandas.core.groupby import DataFrameGroupBy
import functools
from functools import cached_property
from abc import ABC, abstractmethod
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from geopandas import GeoSeries
from shapely.geometry.base import BaseGeometry

from .bbox import BBox
from .compare import CallableCompare
from .data import DescriptorData, DecoratorData, column
from .resource_ import DescriptorResource
from .preprocess import CallablePreprocessor
from .aggregate import DescriptorAggregate
from .groups import DescriptorGroups
from .column import DescriptorColumns

warnings.filterwarnings('ignore', message='.*initial implementation of Parquet.*')


class Source(ABC):
    resource: DescriptorResource
    compare = CallableCompare()
    data = DescriptorData()
    aggregate = DescriptorAggregate()
    groups = DescriptorGroups()
    columns = DescriptorColumns()

    @classmethod
    @property
    def link(cls):
        return cls.resource.link

    @classmethod
    @property
    def name(cls):
        return cls.resource.name

    @column(dtype='geometry', crs=3857)
    @abstractmethod
    def geometry(self):
        """The geometry that the data entry represents"""

    @column(dtype='datetime64[ns]')
    @abstractmethod
    def timestamp(self):
        """The time of the data entry"""

    @column(dtype=np.uint32)
    @abstractmethod
    def group(self):
        """The group that the data entry is in; identically grouped entries will be aggregated"""

    @timestamp.aggregate
    @abstractmethod
    def timestamp(self):
        """The timestamp of the relevant aggregated data"""

    @column(dtype='geometry', crs=3857, dependent={'geometry'})
    def centroid(self):
        return self.data.geometry.centroid.values

    @column(dtype='|S14', dependent='centroid')
    def ref(self):
        centroid: GeoSeries = self.data.geometry.centroid.values
        x = centroid.x
        y = centroid.y
        nan = np.isnan(x) | np.isnan(y)
        ref = np.char.join(', ', (x.astype('S6'), y.astype('S6')))
        ref[nan] = ''
        return ref

    @geometry.aggregate
    def geometry(self):
        index = self.groups.index
        count = len(index)
        geometry = np.fromiter((
            GeoSeries.unary_union(group)
            for group in self.groups.dataframes
        ), dtype=object, count=count)
        return GeoSeries(geometry, index=index, crs=4326)

    @centroid.aggregate
    def centroid(self):
        index = self.groups.index
        count = len(index)

    def __contains__(self, item: BBox):
        return item in self.__class__.resource

    def __init__(
            self,
            bbox: BBox,
            *args,
            redo: bool = False,
            **kwargs,
    ):
        self.bbox = bbox
        self.redo = redo
