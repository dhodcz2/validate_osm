import functools
from validate_osm.source.source import BBox
import inspect
import logging
from pathlib import Path
from typing import Union, Iterable, Type, Hashable, Optional, Iterator

import geopandas as gpd
from geopandas import GeoDataFrame
from python_log_indenter import IndentedLoggerAdapter

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
        self._footprint = None
        logger = logging.getLogger(__name__.partition('.')[0])
        logger.setLevel(
            logging.DEBUG if debug else
            logging.INFO if verbose else
            logging.WARNING
        )
        handler = logging.StreamHandler()
        formatter = logging.Formatter(f"%(levelname)10s %(message)s")
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
            self._redo = {}
        elif isinstance(names, str):
            self._redo = frozenset((names,))
        elif isinstance(names, Iterable):
            self._redo = frozenset(names)
        elif names is True:
            self._redo = {'data', 'footprint', 'aggregate', *self.sources.keys()}
        else:
            raise TypeError(names)
        for name, source in self.sources.items():
            source.ignore_file = name in self._redo

    @redo.deleter
    def redo(self):
        self._redo = {}

    @functools.cached_property
    def footprint(self) -> CallableFootprint:
        sources = self.sources.values()
        footprints: Iterator[tuple[Type[CallableFootprint], Source]] = zip(
            (source.footprint for source in sources), sources
        )
        footprint, source = next(footprints)
        for other_footprint, other_source in footprints:
            if other_footprint is not footprint:
                raise ValueError(f"{source.__class__.__name__}.footprint!={other_source.__class__.name}.footprint")

        return footprint(self)

    @property
    def footprints(self) -> GeoDataFrame:
        return self.footprint.footprints

    @footprints.deleter
    def footprints(self):
        del self.footprint.footprints

    @functools.cached_property
    def name(self) -> str:
        names = list(self.sources.keys())
        names.sort()
        return '_'.join(names)

    def __repr__(self):
        return f'{self.__class__.__name__}{self.name}'


    # def __getitem__(self, items) -> 'Compare':
    #     if not isinstance(items, tuple):
    #         items = (items,)
    #     agg = self.aggregate
    #     names = [
    #         item for item in items
    #         if isinstance(item, str)
    #     ]
    #
    #
    #     for item in items:
    #
    #         if isinstance(item, BBox):
    #             ...
    #         elif is
    #



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

    def percent_overlap(self, name: Hashable, others: Optional[Hashable] = None) -> GeoDataFrame:
        """
        Returns a GeoDataFrame where intersection = % overlap of the name's entries compared with others
        :param name:
        :param others:
        :return:
        """
        agg = self.aggregate
        this: GeoDataFrame = agg.xs(name, level='name')
        this['area'] = this.area
        this['centroid'] = this['centroid'].to_crs(4326)
        if others is None:
            others: GeoDataFrame = agg[agg.index.get_level_values('name') != name]
        else:
            others: GeoDataFrame = agg[agg.index.get_level_values('name').isin(others)]

        def gen():
            for row, ubid in zip(this.itertuples(), this.index.get_level_values('ubid')):
                try:
                    other = others.xs(ubid, level='ubid')
                except KeyError:
                    intersection = 0
                else:
                    intersection = row.geometry.intersection(other.geometry.unary_union).area / row.area
                yield (row.geometry, row.centroid, intersection, row.iloc)

        gdf = gpd.GeoDataFrame(gen(), columns=['geometry', 'centroid', 'intersection', 'iloc'])
        return gdf

        """
        
        :param name: Name of the Source that is tested for overlap with others 
        :return: 
        """


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
