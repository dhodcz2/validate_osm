import functools
import inspect
import logging
from pathlib import Path
from typing import Union, Iterable, Type, Iterator, Optional

import numpy as np
import pandas as pd
import shapely.geometry.base
from geopandas import GeoDataFrame
from geopandas import GeoSeries
from python_log_indenter import IndentedLoggerAdapter

from validate_osm.compare.aggregate import DescriptorAggregate
from validate_osm.compare.data import DescriptorData
from validate_osm.compare.plot import DescriptorPlot
from validate_osm.compare.validate import DescriptorValidate
from validate_osm.source.footprint import CallableFootprint
from validate_osm.source.source import (
    Source, BBox
)


class Compare:
    data = DescriptorData()
    aggregate = DescriptorAggregate()
    plot = DescriptorPlot()
    batch: Union[GeoDataFrame]
    sources: dict[str, Source]
    validate = DescriptorValidate()

    # TODO: How best can the user specify which files are to be redone?
    def __init__(
            self,
            bbox: BBox,
            *sources: Type[Source] | Iterable[Type[Source]],
            redo: Union[None, str, Iterable, bool] = None,
            debug: bool = False,
            verbose: bool = False,
            serialize: bool = True
    ):
        if isinstance(sources, Source):
            sources = (sources,)
        elif isinstance(sources, Iterable):
            pass
        else:
            raise TypeError(sources)

        self.sources: dict[str, Source] = {
            source.name: source()
            for source in sources
        }

        if bbox is not None:
            for source in self.sources.values():
                source.bbox = bbox
        self.bbox = bbox
        logger = logging.getLogger(__name__.partition('.')[0])
        logger.setLevel(
            logging.DEBUG if debug else
            logging.INFO if verbose else
            logging.WARNING
        )
        for handler in logger.handlers:
            if isinstance(handler, logging.StreamHandler):
                break
        else:
            handler = logging.StreamHandler()
            formatter = logging.Formatter(f"%(levelname)-10s %(message)s")
            handler.setFormatter(formatter)
            logger.addHandler(handler)
        logger = IndentedLoggerAdapter(logger)
        self.logger = logger
        for source in self.sources.values():
            source.logger = self.logger
        self.redo = redo
        self.serialize = serialize

    @property
    def redo(self):
        return self._redo

    @redo.setter
    def redo(self, names: Union[None, str, Iterable]):
        if names is None or names is False:
            self._redo = frozenset()
        elif isinstance(names, str):
            self._redo = frozenset((names,))
        elif isinstance(names, Iterable):
            self._redo = frozenset(names)
        elif names is True:
            self._redo = frozenset(('data', 'footprint', 'aggregate', *self.sources.keys()))
        else:
            raise TypeError(names)
        if 'sources' in self._redo or 'source' in self._redo:
            self._redo = frozenset((*self.sources.keys(), *self._redo))
        for name, source in self.sources.items():
            if name in self._redo:
                self._redo = frozenset(('data', 'aggregate', *self._redo))
                source.redo = True
            else:
                source.redo = False

    @redo.deleter
    def redo(self):
        self._redo = frozenset()

    @property
    def footprint(self) -> CallableFootprint:
        if hasattr(self, '_footprint'):
            return self._footprint
        sources = self.sources.values()
        footprints: Iterator[tuple[Type[CallableFootprint], Source]] = zip(
            (source.footprint for source in sources), sources
        )
        footprint, source = next(footprints)
        for other_footprint, other_source in footprints:
            if other_footprint is not footprint:
                raise ValueError(f"{source.__class__.__name__}.footprint!={other_source.__class__.name}.footprint")
        self._footprint = footprint = footprint(self)
        return footprint

    @property
    def footprints(self) -> GeoDataFrame:
        return self.footprint.gdf

    @property
    def index(self) -> int | str:
        # TODO: Dynamically determine from Sources
        return 'ubid'

    def _get_index_values(self, gdf: Union[GeoDataFrame, GeoSeries]) -> Iterable[int | str]:
        return gdf.index.get_level_values(level=self.index) if isinstance(gdf.index, pd.MultiIndex) else gdf.index

    def xs(self, key: Union[int | str, pd.Series, pd.DataFrame]) -> GeoDataFrame:
        if isinstance(key, pd.DataFrame):
            return key
        if isinstance(key, (pd.Series)):
            return GeoDataFrame(key)
        if 'footprint' == key or key == 'footprints':
            return self.footprints
        if key in self.names:
            return self.aggregate.xs(key, level='name', drop_level=False)
        return self.aggregate.xs(key, level=self.index, drop_level=False)

    @functools.cached_property
    def name(self) -> str:
        names = list(self.sources.keys())
        names.sort()
        return '_'.join(names)

    @functools.cached_property
    def names(self) -> set[str]:
        return set(self.sources.keys())

    def __repr__(self):
        return f'{self.__class__.__name__}[{self.name}]'

    def __getitem__(self, items) -> 'Compare':
        if not isinstance(items, tuple):
            items = (items,)
        data = self.data
        agg = self.aggregate

        # # [[94.1, -70, 94.8, -72]] or [BBox(94.1, -70, 94.8, -72)]
        try:
            bbox = next(
                item for item in items
                if isinstance(item, (list, BBox))
            )
        except StopIteration:
            bbox = self.bbox
            footprint = self.footprint
        else:
            if isinstance(bbox, list):
                # Flipped because it's typically constructed with ellipsoidal
                projected = BBox(bbox, crs='3857')
                polygon = projected.ellipsoidal
            elif isinstance(bbox, BBox):
                projected = bbox.to_crs(3857)
                # TODO: Why is it ellipsoidal and not cartesian? Something doesn't seem right.
                polygon = projected.ellipsoidal
                # The issue of flipped coords with ellipsoidal/cartesian is very annoying

            footprint = self.footprint[polygon]
            identifiers = set(footprint.gdf.index.get_level_values('ubid'))
            data = data[data.index.get_level_values('ubid').isin(identifiers)]
            agg = agg[agg.index.get_level_values('ubid').isin(identifiers)]

        # ['osm', 'msbf']
        names = {
            item for item in items
            if isinstance(item, str)
        }
        if names:
            data = data[data.index.get_level_values('name').isin(names)]
            agg = data[data.index.get_level_values('name').isin(names)]
            sources = {
                name: source
                for name, source in self.sources.items()
                if name in names
            }
        else:
            sources = self.sources

        data['iloc'] = pd.Series(range(len(data)), dtype='int32')
        agg['iloc'] = pd.Series(range(len(agg)), dtype='int32')

        compare = Compare(
            bbox,
            redo=None,
            debug=False,
            verbose=False,
            serialize=False
        )
        compare.data = data
        compare._footprint = footprint
        compare.aggregate = agg
        # Perhaps create a new Source instance that encapsulates a smaller .data
        compare.sources = sources
        return compare

    @functools.cached_property
    def directory(self) -> Path:
        return Path(inspect.getfile(self.__class__)).parent / self.name

    def matched(self, name: Union[None, int | str, Iterable[int | str]] = None) -> GeoDataFrame:
        """
        Returns compare.aggregate where aggregate.name has matching ubid
        :param name:
        :return:
        """
        agg = self.aggregate
        if name is None:
            # Return where only one entry for UBID
            singles = (
                group for group in
                agg.groupby('ubid').indices.values()
                if len(group) == 1
            )
            return agg.iloc[singles]
        elif isinstance(name, int | str):
            others: GeoDataFrame = agg[agg.index.get_level_values('name') != name]
            agg: GeoDataFrame = agg[agg.index.get_level_values('name') == name]
        elif isinstance(name, Iterable):
            names = set(name)
            others: GeoDataFrame = agg[~agg.index.get_level_values('name').isin(names)]
            agg: GeoDataFrame = agg[agg.index.get_level_values('name').isin(names)]
        else:
            raise TypeError(name)

        ubids = set(agg.index.get_level_values('ubid').intersection(others.index.get_level_values('ubid')))
        return agg[agg.index.get_level_values('ubid').isin(ubids)]

    def unmatched(self, name: Optional[int | str]):
        """
        Returns compare.aggregate where aggregate.name has no matching ubid
        :param name:
        :return:
        """
        agg = self.aggregate
        if name is None:
            # Return where only one entry for UBID
            singles = (
                group for group in
                agg.groupby('ubid').indices.values()
                if len(group) == 1
            )
            return agg.iloc[singles]
        elif isinstance(name, int | str):
            others: GeoDataFrame = agg[agg.index.get_level_values('name') != name]
            agg: GeoDataFrame = agg[agg.index.get_level_values('name') == name]
        elif isinstance(name, Iterable):
            names = set(name)
            others: GeoDataFrame = agg[~agg.index.get_level_values('name').isin(names)]
            agg: GeoDataFrame = agg[agg.index.get_level_values('name').isin(names)]
        else:
            raise TypeError(name)

        ubids = set(agg.index.get_level_values('ubid').difference(others.index.get_level_values('ubid')))
        return agg[agg.index.get_level_values('ubid').isin(ubids)]

    def match_rate(self, name: Union[int | str, Iterable[int | str],]) -> float:
        if isinstance(name, int | str):
            agg = self.aggregate.xs(name, level='name')
            return len(self.matched(name)) / len(agg)
        elif isinstance(name, Iterable):
            names = set(name)
            agg = self.aggregate
            agg = agg[agg.index.get_level_values('ubid').isin(names)]
            return len(self.matched(names)) / len(agg)
        else:
            raise TypeError(name)

    def containment(self, of, in_) -> pd.Series:
        of: GeoDataFrame = self.xs(of)
        in_: GeoDataFrame = self.xs(in_)

        area: dict[int | str, float] = {
            ubid: area
            for ubid, area in zip(self._get_index_values(of), of.area)
        }
        in_: dict[int | str, shapely.geometry.base.BaseGeometry] = {
            i: geom
            for i, geom in zip(self._get_index_values(in_), in_.geometry)
        }
        of: dict[int | str, shapely.geometry.base.BaseGeometry] = {
            i: geom
            for i, geom in zip(self._get_index_values(of), of.geometry)
            if i in in_
        }
        return pd.Series((
            g.intersection(in_[i]).area / area[i]
            for i, g in of.items()
        ), index=pd.Index(of.keys(), name=self.index), dtype='float64')

    def difference_percent(self, of, and_, value: int | str) -> pd.Series:
        of = self.xs(of)
        of = of[of[value].notna()]
        and_ = self.xs(and_)
        and_ = and_[and_[value].notna()]

        of = {
            i: val
            for i, val in zip(self._get_index_values(of), of[value])
        }
        and_ = {
            i: val
            for i, val in zip(self._get_index_values(and_), and_[value])
            if i in of
        }

        def gen():
            for i, a in and_.items():
                b = of[i]
                if a > b:
                    yield (a - b) / a
                else:
                    yield (b - a) / b

        return pd.Series(gen(), index=pd.Index(and_.keys(), name=self.index), dtype='float64')

    def difference_absolute(self, of, and_, value: int | str) -> pd.Series:
        of: GeoDataFrame = self.xs(of)
        of: GeoDataFrame = of[of[value].notna()]
        and_: GeoDataFrame = self.xs(and_)
        and_: GeoDataFrame = and_[and_[value].notna()]

        of: dict[int | str, float] = {
            i: val
            for i, val in zip(self._get_index_values(of), of[value])
        }
        and_: dict[int | str, float] = {
            i: val
            for i, val in zip(self._get_index_values(and_), and_[value])
            if i in of
        }

        def gen():
            for i, a in and_.item():
                b = of[i]
                if a > b:
                    yield a - b
                else:
                    yield b - a

        return pd.Series(gen(), index=pd.Index(and_.keys(), name=self.index), dtype='float64')

    def difference_scaled(self, of, and_, value: int | str) -> pd.Series:
        difference = self.difference_percent(of, and_, value)
        containment = self.containment(of, and_)
        result = difference * containment
        result = result[result.notna()]
        return result


    def intersection(self, of, and_) -> GeoSeries:
        of: GeoDataFrame = self.xs(of)
        and_: GeoDataFrame = self.xs(and_)
        crs = of.crs
        of: dict[str, shapely.geometry.base.BaseGeometry] = {
            ubid: geom
            for ubid, geom in zip(self._get_index_values(of), of.geometry)
        }
        and_: dict[str, shapely.geometry.base.BaseGeometry] = {
            ubid: geom
            for ubid, geom in zip(self._get_index_values(and_), and_.geometry)
            if ubid in of
        }

        return GeoSeries((
            geom.intersection(of[i])
            for i, geom in and_.items()
        ), index=pd.Index(and_.keys(), name=self.index), crs=crs, name='geometry')

    def union(self, of, and_) -> GeoSeries:
        of = self.xs(of)
        and_ = self.xs(and_)
        crs = of.crs
        of: dict[str, shapely.geometry.base.BaseGeometry] = {
            i: geom
            for i, geom in zip(self._get_index_values(of), of.geometry)
        }
        and_: dict[str, shapely.geometry.base.BaseGeometry] = {
            i: geom
            for i, geom in zip(self._get_index_values(and_), and_.geometry)
            if i in of
        }

        return GeoSeries((
            geom.union(of[i])
            for i, geom in and_.items()
        ), index=pd.Index(and_.keys(), name=self.index), crs=crs, name='geometry')

    def quality(self, of, and_) -> pd.Series:
        union = self.union(of, and_)
        intersection = self.intersection(of, and_)
        return self.containment(of=intersection, in_=union)
