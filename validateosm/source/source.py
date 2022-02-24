import geopandas as gpd
import shapely.ops

import inspect

import numpy.typing
from pathlib import Path

import abc
import copy
import dataclasses
import datetime
import functools
import itertools
from typing import Any, Container, Type
from typing import Optional, Iterable, Iterator, Collection, Union
from typing import ValuesView
import numpy
import numpy as np
import pandas as pd
import pyproj
import shapely.geometry.base
from annoy import AnnoyIndex
from geopandas import GeoSeries, GeoDataFrame
from pandas import Series, DataFrame

from validateosm.source.static import StaticRegional, StaticBase, StaticNaive
from validateosm.source.aggregate import DescriptorAggregate, DecoratorAggregate
from validateosm.source.data import DecoratorData, DescriptorData
from validateosm.source.groups import (
    DescriptorGroup,
    DecoratorGroup,
    Groups
)


@dataclasses.dataclass
class BBoxStruct:
    ellipsoidal: shapely.geometry.Polygon
    cartesian: shapely.geometry.Polygon
    crs: Any

    # _crs: Any = dataclasses.field(init=False, repr=False)

    # def __post_init__(self):
    #     self._flipped = False

    def __repr__(self):
        return str([
            round(bound, 2)
            for bound in self.ellipsoidal.bounds
        ])

    # @property
    # def flipped(self):
    #     return self._flipped
    #
    # @flipped.setter
    # def flipped(self, val: bool):
    #     match val:
    #         case bool():
    #             if self.flipped != val:
    #                 self._flipped = val
    #                 self.ellipsoidal = shapely.geometry.Polygon((y, x) for x, y in self.ellipsoidal)
    #                 self.cartesian = shapely.geometry.Polygon((y, x) for x, y in self.cartesian)
    #         case _:
    #             raise TypeError(val)

    # @property
    # def crs(self):
    #     return self._crs
    #
    # @crs.setter
    # def crs(self, val: Any):
    #     if hasattr(self, '_crs'):
    #         geom = gpd.GeoSeries((self.ellipsoidal, self.cartesian), crs=self.crs)
    #         self._crs = val
    #         self.ellipsoidal, self.cartesian = geom.to_crs(val)
    #     else:
    #         self._crs = val
    #


# TODO: BBox understands its recipient; if

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

    @property
    def resource(self) -> BBoxStruct:
        if self._owner not in self._cache_static:
            d = self.data
            gs = gpd.GeoSeries((d.ellipsoidal, d.cartesian), crs=d.crs)
            static = self._owner.resource
            if static.flipped:
                cartesian, ellipsoidal = gs.to_crs(static.crs)
            else:
                ellipsoidal, cartesian = gs.to_crs(static.crs)

            # I was considering flipping the coords, but maybe just changing ellipsoidal with cartesian is ie
            # gs = gs.map(lambda geom: shapely.ops.transform(lambda x, y, z=None: (y, x, z), geom))
            bbox = BBoxStruct(ellipsoidal, cartesian, crs=d.crs)
            self._cache_static[self._owner] = bbox
            return bbox
        else:
            return self._cache_static[self._owner]


class Source(abc.ABC):
    '''
    raw >> data >> groups >> aggregate >> identity >> exclude >> batch

    pipeline methods except for raw extraction methods are defined with _ and decorated with the name to minimize
    namespace clutter
    '''

    resource: Union[gpd.GeoDataFrame, StaticBase]
    data: Union[DescriptorData, GeoDataFrame] = DescriptorData()
    groups: Union[Groups, DescriptorGroup] = DescriptorGroup()
    aggregate: Union[DescriptorAggregate, GeoDataFrame] = DescriptorAggregate()
    footprint: Optional['Source']
    name: str
    link: str
    bbox: BBox
    resource: Union[StaticBase, pd.DataFrame, gpd.GeoDataFrame]

    def resource(self) -> Union[StaticBase]:
        """An instance or Iterator of instances that encapsulate the raw data that is entering this pipeline."""

    resource = (property(abc.abstractmethod(resource)))

    def identity(self) -> Optional[Series]:
        """ Iterates across Source.aggregate and yields keys that will determine the index of Source.batch"""

    def exclude(self) -> Optional[numpy.typing.NDArray[bool]]:
        """ Iterates across Source.aggregate and yields True if entry is to be excluded from Source.batch """

    def footprint(cls) -> Optional['Source']:
        """
        If defined, Source will reference footprint's geometry to determine its aggregate, rather than self-referential
        containment.
        """
        return None

    footprint = classmethod(property(footprint))

    def name(cls) -> str:
        """A short, abbreviated name that may be used for quickly selecting a specific source."""

    name = classmethod(property(abc.abstractmethod(name)))

    def link(cls) -> str:
        """A link to the page for further data regarding the Source"""

    link = classmethod(property(abc.abstractmethod(link)))

    def bbox(cls) -> BBox:
        """A BBox which represents the bounds of which relevant data from the Source is extracted."""

    bbox = classmethod(property(abc.abstractmethod(bbox)))

    @classmethod
    @property
    def validating(cls) -> set[str]:
        """The specific columns that will be validated"""
        return cls.data.validating

    @classmethod
    @property
    def identifier(cls) -> set[str]:
        """The identifier(s) that may be used to establish help establish relationships
         both between and across datasets"""
        return cls.data.identifier

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

    @DecoratorAggregate('geometry')
    def _(self) -> GeoSeries:
        single = self.groups.ungrouped['geometry']
        multi = (
            gdf['geometry'].unary_union
            for gdf in self.groups.grouped
        )
        return GeoSeries(
            data=itertools.chain(single, multi),
            crs=self.groups.ungrouped['geometry'].crs,
        ).to_crs(4326)

    @DecoratorAggregate('centroid')
    def _(self) -> GeoSeries:
        single = self.groups.ungrouped['centroid']
        multi = (
            gdf['centroid'].unary_union
            for gdf in self.groups.grouped
        )
        return GeoSeries(
            data=itertools.chain(single, multi),
            crs=self.groups.ungrouped['centroid'].crs,
        ).to_crs(3857)

    @DecoratorAggregate('ref')
    def _(self) -> GeoSeries:
        single = self.groups.ungrouped['ref']
        multi = (
            gdf['ref'].unary_union
            for gdf in self.groups.grouped
        )
        multi = (
            None if union is None
            else union.centroid
            for union in multi
        )
        return GeoSeries(
            data=itertools.chain(single, multi),
            crs=self.groups.ungrouped['ref'].crs,
        )

    @DecoratorAggregate('data')
    def _(self) -> Series:
        single = (
            [i]
            for i in self.groups.ungrouped.index
        )
        multi = (
            list(gdf.index)
            for gdf in self.groups.grouped
        )
        return Series(data=itertools.chain(single, multi), )

    # directory: Path
    #
    # def directory(cls) -> Path:
    #     return Path(inspect.getfile(cls)).parent / 'resources' / cls.__name__
    #
    # directory = classmethod(property(directory))

    @DecoratorGroup(name='footprint')
    def _(self) -> ValuesView[Collection[int]]:
        data: gpd.GeoDataFrame = self.data[['centroid', 'geometry']]
        data['geometry'] = data['geometry'].to_crs(3857)
        data['area'] = data.area

        if self.footprint is not None and self.footprint is not self.__class__:
            # Use external footprint
            footprint = pd.Series(index=data.index)
            try:
                external_footprint = self.footprint._footprint
                annoy = self.footprint._annoy
            except AttributeError:
                external_footprint: gpd.GeoDataFrame = self.footprint.aggregate[['geometry', 'centroid']]
                external_footprint['geometry'] = external_footprint.to_crs(3857)
                annoy = AnnoyIndex(2, 'euclidean')
                for i, centroid in enumerate(external_footprint['centroid']):
                    annoy.add_item(i, (centroid.x, centroid.y))
                annoy.build(10)
                # TODO: Does it cause memory issues to store annoy and footprint in the Source class for reuse?
                self.footprint._annoy = annoy
                self.footprint._footprint = external_footprint

            for i, (c, a, g) in enumerate(data[['centroid', 'area', 'geometry']].values):
                for n in annoy.get_nns_by_vector((c.x, c.y), 5):
                    external = external_footprint.iloc[n]
                    if not external['geometry'].intersects(g):
                        continue
                    if external['geometry'].intersection(g).area / a < .5:
                        continue
                    footprint.iloc[i] = n
                    break

            # Exclude anything that is not encapsulated by the external footprint
            footprint = footprint[footprint.duplicated(keep=False)]


        else:
            # Use internal footprint
            data['buffer'] = data['geometry'].buffer(1)
            data.sort_values(by='area', ascending=False)
            footprints = pd.Series(range(len(data)), index=data.index)

            annoy = AnnoyIndex(2, 'euclidean')
            for i, centroid in enumerate(data['centroid']):
                annoy.add_item(i, (centroid.x, centroid.y))  # lower i means higher area
            annoy.build(10)

            for i, (g, a) in enumerate(data[['geometry', 'area']].values):
                for n in annoy.get_nns_by_item(i, 10):
                    if i < n:  # If the neighbor is smaller
                        continue
                    footprint = data.iloc[n]
                    if not footprint['buffer'].contains(g):  # If footprint doesn't fully contain g
                        continue
                    # Because we are descending in size, we know that all entries will have the index
                    # of the largest segment of the group
                    footprints.iloc[i] = footprints.iloc[n]
                    break

        footprints = footprints[self.data.index]  # Retain original order because groupby.indices returns iloc
        groups = footprints.groupby(footprints, dropna=True).indices.values()
        return groups


{
}
