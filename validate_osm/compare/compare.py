import functools
import inspect
import logging
from pathlib import Path
from typing import Union, Iterable, Type, Hashable, Optional, Iterator

import geopandas as gpd
import numpy as np
import pandas as pd
from geopandas import GeoDataFrame
from python_log_indenter import IndentedLoggerAdapter

from validate_osm.compare.validate import DescriptorValidate
from validate_osm.compare.aggregate import DescriptorAggregate
from validate_osm.compare.data import DescriptorData
from validate_osm.compare.plot import DescriptorPlot
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

    @functools.cached_property
    def name(self) -> str:
        names = list(self.sources.keys())
        names.sort()
        return '_'.join(names)

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

    def matched(self, name: Union[None, Hashable, Iterable[Hashable]] = None) -> GeoDataFrame:
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
        elif isinstance(name, Hashable):
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

    def unmatched(self, name: Optional[Hashable]):
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
        elif isinstance(name, Hashable):
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

    def match_rate(self, name: Union[Hashable, Iterable[Hashable],]) -> float:
        if isinstance(name, Hashable):
            agg = self.aggregate.xs(name, level='name')
            return len(self.matched(name)) / len(agg)
        elif isinstance(name, Iterable):
            names = set(name)
            agg = self.aggregate
            agg = agg[agg.index.get_level_values('ubid').isin(names)]
            return len(self.matched(names)) / len(agg)
        else:
            raise TypeError(name)

    def containment(self, of: GeoDataFrame, within: GeoDataFrame) -> pd.Series:
        geometries = {
            ubid: geom
            for ubid, geom in zip(within.index, within['geometry'])
        }

        def gen():
            for ubid, g, a, in zip(of.index, of['geometry'], of.area):
                if ubid in geometries:
                    yield g.intersection(geometries[ubid]).area / a
                else:
                    yield np.nan

        return pd.Series(gen(), index=of.index, dtype='float64')

    def completion(self, of: GeoDataFrame) -> pd.Series:
        overlap = self.containment(of=self.footprints, within=of)  # of=footprints is not a mistake
        overlaps = {
            ubid: overlap
            for ubid, overlap in zip(overlap.index, overlap)
        }

        return pd.Series((
            overlaps.get(ubid, np.nan)
            for ubid in of.index
        ), index=of.index, dtype='float64')

    def percent_difference(self, of: GeoDataFrame, according_to: GeoDataFrame, regarding: Hashable):
        values = {
            ubid: value
            for ubid, value in zip(according_to.index, according_to[regarding])
        }

        return pd.Series((

        ))

    def scaled_percent_difference(self):
        raise NotImplementedError


"""
1.  Compare.data
    data
    with self:
        with compare.footprint as footprint:
            data = footprint(data)
        self._data = data
        
    with compare.footprint as footprint:
        return footprint(data)
        
2.  Compare.footprint
        compare.data
"""
