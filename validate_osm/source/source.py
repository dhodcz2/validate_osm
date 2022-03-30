import abc
import itertools
import logging
import warnings
from typing import Type
from typing import Optional, Union

import geopandas as gpd
import numpy as np
import numpy.typing
import pandas as pd
from geopandas import GeoDataFrame

from validate_osm.source.bbox import BBox
from validate_osm.source.aggregate import FactoryAggregate
from validate_osm.source.data import DecoratorData, DescriptorData
from validate_osm.source.footprint import CallableFootprint
from validate_osm.source.resource import StaticBase

warnings.filterwarnings('ignore', message='.*initial implementation of Parquet.*')


class SourceMeta(abc.ABCMeta, type):
    def __new__(cls, name, bases, local):
        object = super(SourceMeta, cls).__new__(cls, name, bases, local)
        if '_data' not in local:
            setattr(object, '_data', {})
        return object


class Source(abc.ABC, metaclass=SourceMeta):
    def __init__(self, redo=False):
        abstracts = [
            struct
            for struct in self.__class__.data.structs.values()
            if struct.abstract
        ]
        if abstracts:
            raise TypeError(f"{self.__class__.__name__} inherited abstract methods for its data: {abstracts}")
        self.redo = redo

        self.logger = logging.getLogger(__name__.partition('.')[0])
        # TODO: How can Source add a handler to its own instance of logger without things getting crazy?
        # self.logger.setLevel(logging.INFO)
        # handler = logging.StreamHandler()
        # formatter = logging.Formatter(f"%(levelname)10s %(message)s")
        # handler.setFormatter(formatter)
        # self.logger.addHandler(handler)

    def __contains__(self, item: BBox) -> bool:
        return item in self.__class__.resource

    # def __eq__(self, other):
    #     TODO: True if the sources have implemented all of the same data methods
    #     raise NotImplementedError

    '''
    raw >> data >> groups >> aggregate >> identity >> exclude >> batch

    pipeline methods except for raw extraction methods are defined with _ and decorated with the name to minimize
    namespace clutter
    '''

    resource: Union[gpd.GeoDataFrame, StaticBase]
    data: Union[DescriptorData, GeoDataFrame] = DescriptorData()
    footprint: Type[CallableFootprint] = CallableFootprint
    # groups: Union[Groups, DescriptorGroup] = DescriptorGroup()
    aggregate_factory: Type[FactoryAggregate] = FactoryAggregate
    name: str
    link: str
    resource: Union[StaticBase, pd.DataFrame, gpd.GeoDataFrame]
    ignore_file = False

    @property
    def redo(self):
        return self._ignore_file

    @redo.setter
    def redo(self, val: bool):
        if not isinstance(val, bool):
            raise TypeError(val)
        self._ignore_file = val

    @redo.deleter
    def redo(self):
        self._ignore_file = False

    @classmethod
    @property
    def name(cls) -> str:
        return cls.resource.name

    @classmethod
    @property
    def link(cls) -> str:
        return cls.resource.link

    def group(self) -> GeoDataFrame:
        """
        Assign to self.data an index
        :return: None
        """
        data = self.data
        return data.set_index(pd.Index(data=itertools.repeat(np.nan, len(data)), name='group'), append=True)

    def resource(self) -> Union[StaticBase]:
        """An instance or Iterator of instances that encapsulate the raw data that is entering this pipeline."""

    resource = (property(abc.abstractmethod(resource)))

    def exclude(self) -> Optional[numpy.typing.NDArray[bool]]:
        """ Iterates across Source.aggregate and yields True if entry is to be excluded from Source.batch """

    @classmethod
    @property
    def validating(cls) -> set[str]:
        """The specific columns that will be validated"""
        return cls.data.validating

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
        return (
            self.data
                .loc[self.data['geometry'].notna(), 'geometry']
                .to_crs(3857)
                .centroid
        )

    @DecoratorData(dtype='string', crs=None, dependent='centroid')
    def ref(self):
        return pd.Series((
            f'{centroid.y:.4f}, {centroid.x:.4f}'
            if centroid is not None
            else None
            for centroid in self.data['centroid'].to_crs(4326)
        ), index=self.data.index, dtype='string')

        # loc = self.data['centroid'].notna()
        # return GeoSeries((
        #     shapely.geometry.Point(centroid.y, centroid.x)
        #     for centroid in self.data.loc[loc, 'centroid'].to_crs(4326)
        # ), index=self.data.loc[loc].index)
