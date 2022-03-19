import functools
import inspect
import logging
import sys
from pathlib import Path
from typing import Union, Iterable, Type, Hashable, Optional

import geopandas as gpd
from geopandas import GeoDataFrame

from validate_osm.compare.aggregate import DescriptorAggregate
from validate_osm.compare.data import DescriptorData
from validate_osm.compare.plot import DescriptorPlot
from validate_osm.source.footprint import CallableFootprint
from validate_osm.source.source import (
    Source, BBox
)
from validate_osm.args import global_args


class Compare:
    data = DescriptorData()
    aggregate = DescriptorAggregate()
    plot = DescriptorPlot()
    batch: Union[GeoDataFrame]
    sources: dict[str, Source]

    def __init__(
            self,
            bbox: BBox,
            *sources: Type[Source] | Iterable[Type[Source]],
            ignore_file=False,
            debug: bool = False,
            verbose: bool = False,
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
        self.ignore_file = ignore_file

        if bbox is not None:
            for source in self.sources.values():
                source.bbox = bbox
        self.bbox = bbox
        self._footprint = None
        logging.basicConfig(
            stream=sys.stdout,
            level=(
                logging.DEBUG if debug else
                logging.INFO if verbose else
                logging.WARNING
            ),
            format='%(asctime)s - %(name)s - %(levelname)'
        )

    @property
    def footprint(self) -> CallableFootprint:
        if self._footprint is None:
            raise RuntimeError(f"Compare.footprint is dependent upon Compare.data; first, instantiate Compare.data")
        return self._footprint

    @footprint.setter
    def footprint(self, value):
        self._footprint = value

    @footprint.deleter
    def footprint(self):
        logging.warning(f'footprints cannot be regenerated until compare.data is regenerated')
        del self._footprint

    @functools.cached_property
    def names(self) -> list[str]:
        names = list(self.sources.keys())
        names.sort()
        return names

    def __repr__(self):
        return f'{self.__class__.__name__}{self.names}'

    def __getitem__(self, item: str) -> Source:
        return self.sources[item]

    @functools.cached_property
    def directory(self) -> Path:
        return Path(inspect.getfile(self.__class__)).parent / '_'.join(self.names)

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

    def overlap(self, name: Hashable, others: Optional[Hashable] = None) -> GeoDataFrame:
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
