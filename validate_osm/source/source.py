import logging
import abc
import dataclasses
import itertools
import warnings
from typing import Any, Container, Type
from typing import Optional, Collection, Union

import geopandas as gpd
import numpy as np
import numpy.typing
import pandas as pd
import shapely.geometry.base
import shapely.ops
from geopandas import GeoSeries, GeoDataFrame

from validate_osm.source.aggregate import AggregateFactory
from validate_osm.source.data import DecoratorData, DescriptorData
from validate_osm.source.footprint import CallableFootprint
from validate_osm.source.resource import StaticBase

warnings.filterwarnings('ignore', message='.*initial implementation of Parquet.*')


@dataclasses.dataclass
class BBoxStruct:
    ellipsoidal: shapely.geometry.Polygon
    cartesian: shapely.geometry.Polygon
    crs: Any

    def __repr__(self):
        return str([
            round(bound, 2)
            for bound in self.ellipsoidal.bounds
        ])

    def __str__(self):
        return '_'.join(
            str(round(bound, 5))
            for bound in self.ellipsoidal.bounds
        )


class BBox:
    _cache_static: dict[type, BBoxStruct] = {}

    def __init__(self, ellipsoidal_4326_bbox: Union[Collection[float], shapely.geometry.Polygon]):
        if isinstance(ellipsoidal_4326_bbox, shapely.geometry.Polygon):
            ellipsoidal = ellipsoidal_4326_bbox
        elif isinstance(ellipsoidal_4326_bbox, Container):
            e = ellipsoidal_4326_bbox
            minx = min(e[0], e[2])
            maxx = max(e[0], e[2])
            miny = min(e[1], e[3])
            maxy = max(e[1], e[3])
            ellipsoidal = shapely.geometry.Polygon((
                (minx, miny), (maxx, miny), (maxx, maxy), (minx, maxy),
            ))
        else:
            raise TypeError(ellipsoidal_4326_bbox)
        cartesian = shapely.geometry.Polygon(((y, x) for (x, y) in ellipsoidal.exterior.coords))
        self.data = BBoxStruct(ellipsoidal, cartesian, crs='epsg:4326')

    def __get__(self, instance, owner):
        self._instance: Source = instance
        self._owner: Type[Source] = owner
        return self

    def __repr__(self):
        return self.data.__repr__()

    def __str__(self):
        return self.data.__str__()

    # @property
    # def resource(self) -> BBoxStruct:
    #     if self._owner not in self._cache_static:
    #         d = self.data
    #         gs = gpd.GeoSeries((d.ellipsoidal, d.cartesian), crs=d.crs)
    #         # TODO: cannot use self.owner.resource because this instantiates the dataframe
    #         static = self._owner.resource
    #         if static.flipped:
    #             cartesian, ellipsoidal = gs.to_crs(static.crs)
    #         else:
    #             ellipsoidal, cartesian = gs.to_crs(static.crs)
    #
    #         # I was considering flipping the coords, but maybe just changing ellipsoidal with cartesian is ie
    #         # gs = gs.map(lambda geom: shapely.ops.transform(lambda x, y, z=None: (y, x, z), geom))
    #         bbox = BBoxStruct(ellipsoidal, cartesian, crs=d.crs)
    #         self._cache_static[self._owner] = bbox
    #         return bbox
    #     else:
    #         return self._cache_static[self._owner]
    #


class SourceMeta(abc.ABCMeta, type):
    def __new__(cls, name, bases, local):
        object = super(SourceMeta, cls).__new__(cls, name, bases, local)
        if '_data' not in local:
            setattr(object, '_data', {})
        return object


class Source(abc.ABC, metaclass=SourceMeta):
    def __init__(self, ignore_file=False):
        abstracts = [
            struct
            for struct in self.__class__.data.structs.values()
            if struct.abstract
        ]
        if abstracts:
            raise TypeError(f"{self.__class__.__name__} inherited abstract methods for its data: {abstracts}")
        self.ignore_file = ignore_file

        self.logger = logging.getLogger(__name__.partition('.')[0])
        # TODO: How can Source add a handler to its own instance of logger without things getting crazy?
        # self.logger.setLevel(logging.INFO)
        # handler = logging.StreamHandler()
        # formatter = logging.Formatter(f"%(levelname)10s %(message)s")
        # handler.setFormatter(formatter)
        # self.logger.addHandler(handler)

    '''
    raw >> data >> groups >> aggregate >> identity >> exclude >> batch

    pipeline methods except for raw extraction methods are defined with _ and decorated with the name to minimize
    namespace clutter
    '''

    resource: Union[gpd.GeoDataFrame, StaticBase]
    data: Union[DescriptorData, GeoDataFrame] = DescriptorData()
    footprint: Type[CallableFootprint] = CallableFootprint
    # groups: Union[Groups, DescriptorGroup] = DescriptorGroup()
    aggregate_factory: AggregateFactory = AggregateFactory()
    name: str
    link: str
    bbox: BBox
    resource: Union[StaticBase, pd.DataFrame, gpd.GeoDataFrame]
    ignore_file = False

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

    @DecoratorData(dtype='geometry', crs=4326)
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

    @DecoratorData(dtype='geometry', crs=None, dependent='centroid')
    def ref(self):
        loc = self.data['centroid'].notna()
        return GeoSeries((
            shapely.geometry.Point(centroid.y, centroid.x)
            for centroid in self.data.loc[loc, 'centroid']
        ), index=self.data.loc[loc].index)
