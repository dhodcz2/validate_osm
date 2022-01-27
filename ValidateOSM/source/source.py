import copy
import abc
import dataclasses
import datetime
import functools
import itertools
import os
from argparse import ArgumentParser
from pathlib import Path
from typing import Any, Callable
from typing import Optional, Iterable, Iterator, Collection, Union
from typing import ValuesView

import geopandas as gpd
import numpy as np
import pandas as pd
import pyproj
import shapely.geometry.base
from annoy import AnnoyIndex
from geopandas import GeoSeries, GeoDataFrame
from pandas import DataFrame
from pandas import Series

from ValidateOSM.source.aggregate import DescriptorAggregate, DecoratorAggregate
from ValidateOSM.source.data import DecoratorData, DescriptorData
from ValidateOSM.source.groups import (
    DescriptorGroup,
    DecoratorGroup,
)


class _Namespace:
    debug: bool


_argparser = ArgumentParser()
_argparser.add_argument('--debug', dest='debug', action='store_true')
args = _argparser.parse_args(namespace=_Namespace)


@dataclasses.dataclass
class _BBox:
    ellipsoidal: Collection[float]
    cartesian: Collection[float]
    crs: Any
    _flipped: bool = dataclasses.field(init=False, repr=False)

    # _flipped: bool = False

    @property
    def flipped(self):
        return self._flipped

    @flipped.setter
    def flipped(self, val: bool):
        if not isinstance(val, bool):
            raise TypeError(val)
        if self.flipped != val:
            e = self.ellipsoidal
            self.ellipsoidal = (*e[:2:-1], *e[2:4:-1])
            c = self.cartesian
            self.cartesian = (*c[:2:-1], *c[2:4:-1])
            self.flipped = val

    @property
    def crs(self):
        return self._crs

    @crs.setter
    def crs(self, val: Any):
        trans = pyproj.Transformer.from_crs(self.crs, val)
        self.ellipsoidal = (
            *trans.transform(*self.ellipsoidal[:2]),
            *trans.transform(*self.ellipsoidal[2:4])
        )
        self.cartesian = (
            *trans.transform(*self.cartesian[:2]),
            *trans.transform(*self.cartesian[2:4])
        )
        self._crs = val


class BBox:
    _cache_raw: dict[type, _BBox] = {}
    _cache_data: dict[type, _BBox] = {}

    def __init__(self, ellipsoidal_4326_bbox: Collection[float]):
        self._ellipsoidal = ellipsoidal_4326_bbox
        self._cartesian = (*self._ellipsoidal[:2:-1], *self._ellipsoidal[2:4:-1])

    def __get__(self, instance, owner):
        self._instance = instance
        self._owner = owner
        return self

    @property
    def raw(self):
        if self._instance is None:
            raise RuntimeError(
                f"This should only be called when the class has been instantiated, as the best-scaling way for users "
                f"to modify Source.bbox.raw is to modify within __init__; syntax is less clean when performed in the "
                f"global scope during Class frame construction."
            )
        return self.__class__._cache_raw.setdefault(
            self._owner,
            _BBox(ellipsoidal=self._ellipsoidal, cartesian=self._cartesian, crs='epsg:4326')
        )

    @property
    def data(self):
        return self.__class__._cache_data.setdefault(self._owner, copy.copy(self.raw))


class Source(abc.ABC):
    # TODO: Source.raw will be a descriptor with functionality for automatically
    #   downloading and deleting the raw source file.
    data = DescriptorData()
    groups = DescriptorGroup()
    aggregate = DescriptorAggregate()
    raw: Union[object, Iterator[object]] = None
    bbox: BBox
    path_static: Optional[Path]

    def __new__(cls, **kwargs):
        @property
        def decorator(func: Callable):
            @functools.wraps(func)
            def wrapper(self: Source):
                if not hasattr(self, '_raw'):
                    self._raw = func(self)
                return self._raw

            return wrapper

        # Cache self.raw and apply @property if user has forgotten
        cls.raw = decorator(cls.raw.fget if isinstance(cls.raw, property) else cls.raw)
        instance: Source = super(Source, cls).__new__(**kwargs)
        instance.__init__(**kwargs)
        return instance

    @abc.abstractmethod
    @property
    def raw(self) -> Union[object, Iterator[object]]:
        ...

    raw: object

    def read_file(self, path: Path):
        if not os.path.isdir(path):
            return gpd.read_file(
                filename=path,
                bbox=self.bbox.raw.cartesian,
                rows=100 if args.debug else None,
            )
        else:
            files: Iterator[Path] = (
                p for p in path.glob('**/*')
                if p.is_file()
            )
            return (
                gpd.read_file(
                    filename=f,
                    bbox=self.bbox.raw.cartesian,
                    rows=100 if args.debug else None,
                )
                for f in files
            )

    @staticmethod
    def concat(gdfs: Iterable[GeoDataFrame]) -> GeoDataFrame:
        """Workaround because GeoDataFrame.concat returns DataFrame; we want to preserve CRS."""
        crs = {}

        def generator():
            nonlocal gdfs
            gdfs = iter(gdfs)
            gdf = next(gdfs)
            for col in gdf:
                if not isinstance(gdf[col], GeoSeries):
                    continue
                gs: GeoSeries = gdf[col]
                crs[col] = gs.crs
            yield gdf
            yield from gdfs

        result: DataFrame = pd.concat(generator())
        result: GeoDataFrame = GeoDataFrame({
            col: (
                result[col] if col not in crs
                else GeoSeries(result[col], crs=crs)
            )
            for col in result
        })
        return result

    @classmethod
    @property
    @abc.abstractmethod
    def name(cls) -> str:
        ...

    @classmethod
    @property
    @abc.abstractmethod
    def source_information(cls) -> str:
        ...

    validating = classmethod(property(data.validating))
    identifier: str = classmethod(property(data.identifier))

    @DecoratorData(dtype='geometry', crs=4326)
    @abc.abstractmethod
    def geometry(self) -> Iterable[shapely.geometry.base.BaseGeometry]:
        """The geometry that the data entry represents"""

    @DecoratorData(dtype='datetime64[ns]')
    @abc.abstractmethod
    def timestamp(self) -> Iterable[datetime.datetime]:
        """The time of the data entry"""

    @DecoratorData(dtype='geometry', crs=3857, dependent={'geometry'})
    def centroid(self):
        return (
            self.data
                .loc[self.data['geometry'].notna(), 'geometry']
                .to_crs(3857)
                .centroid
        )

    @DecoratorData(dtype='geometry', crs=None)
    def ref(self):
        loc = self.data['centroid'].notna()
        return GeoSeries((
            shapely.geometry.Point(centroid.y, centroid.x)
            for centroid in self.data.loc[loc, 'centroid']
        ), index=self.data.loc[loc].index)

    @DecoratorAggregate('geometry')
    def _(self) -> GeoSeries:
        single = self.groups.single_len['geometry']
        multi = (
            gdf['geometry'].unary_union
            for gdf in self.groups.multi_len
        )
        return GeoSeries(
            data=itertools.chain(single, multi),
            crs=self.groups.single_len['geometry'].crs,
        ).to_crs(4326)

    @DecoratorAggregate('centroid')
    def _(self) -> GeoSeries:
        single = self.groups.single_len['centroid']
        multi = (
            gdf['centroid'].unary_union
            for gdf in self.groups.multi_len
        )
        return GeoSeries(
            data=itertools.chain(single, multi),
            crs=self.groups.single_len['centroid'].crs,
        ).to_crs(3857)

    @DecoratorAggregate('ref')
    def _(self) -> GeoSeries:
        single = self.groups.single_len['ref']
        multi = (
            gdf['ref'].unary_union
            for gdf in self.groups.multi_len
        )
        multi = (
            None if union is None
            else union.centroid
            for union in multi
        )
        return GeoSeries(
            data=itertools.chain(single, multi),
            crs=self.groups.single_len['ref'].crs,
        )

    @DecoratorAggregate('data')
    def _(self) -> Series:
        single = (
            [i]
            for i in self.groups.single_len.index
        )
        multi = (
            list(gdf.index)
            for gdf in self.groups.multi_len
        )
        return Series(data=itertools.chain(single, multi), )

    @DecoratorGroup.dependent(name='containment')
    def _(self, dependency: 'Source') -> ValuesView[Iterable[int]]:
        geometry: GeoSeries = self.data['geometry'].to_crs(3857)
        area = geometry.area
        dep = dependency.aggregate

        try:
            annoy = dependency.annoy
        except AttributeError:
            annoy = dependency.annoy = AnnoyIndex(2, 'euclidean')
            for i, centroid in enumerate(dep['centroid']):
                annoy.add_item(i, (centroid.x, centroid.y))
            annoy.build(10)

        for (c, g, a) in zip(self.data.centroid, geometry, area):
            for i in annoy.get_nns_by_vector((c.x, c.y), 10):
                nearest = dep.loc[i, 'geometry']
                if not nearest.intersects(g):
                    continue
                if nearest.intersection(g).area > a:
                    yield i
                    break
            else:
                return np.nan

    # TODO: DecoratorGroup.dependent should return the iloc of the dependency.aggregate
    # TODO: When I originally implemented this, matching based on identifier didn't seem too helpful;
    #   disconnected buildings within/across datasets would be matched
    # @DecoratorGroup.independent(name='identifier')
    # def _(self, dependency: 'Source') -> ValuesView[Iterable[int]]:
    #     id: str = dependency.identifier
    #     dependency = dependency.loc[dependency[id].notna(), id]
    # reference: dict[int, int] = {
    #     for index, val in zip(dependency.aggregate.index, dependency.aggregate[id])
    #
    # }
#
