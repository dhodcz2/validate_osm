import abc
import warnings
from pathlib import Path

import numpy as np
from geopandas import GeoSeries

from .bbox import BBox
from .compare import CallableCompare
from .data import DescriptorData, DecoratorData
from .resource_ import DescriptorResource
from .preprocess import CallablePreprocessor

warnings.filterwarnings('ignore', message='.*initial implementation of Parquet.*')


class Source(abc.ABC):
    """

    """
    resource: DescriptorResource
    compare = CallableCompare()
    data = DescriptorData()

    @classmethod
    @property
    def link(cls):
        return cls.resource.link

    @classmethod
    @property
    def name(cls):
        return cls.resource.name

    @DecoratorData(dtype='geometry', crs=3857)
    @abc.abstractmethod
    def geometry(self):
        """The geometry that the data entry represents"""

    @DecoratorData(dtype='datetime64[ns]')
    @abc.abstractmethod
    def timestamp(self):
        """The time of the data entry"""

    @DecoratorData(dtype='geometry', crs=3857, dependent={'geometry'})
    def centroid(self):
        return self.data.geometry.centroid.values

    @DecoratorData(dtype='|S14', dependent='centroid')
    def ref(self):
        centroid: GeoSeries = self.data.geometry.centroid.values
        x = centroid.x
        y = centroid.y
        nan = np.isnan(x) | np.isnan(y)
        ref = np.char.join(', ', (x.astype('S6'), y.astype('S6')))
        ref[nan] = ''
        return ref

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
