import functools
import inspect
from pathlib import Path
from typing import Union, Iterable, Type, Generator

import geopandas as gpd
import pandas
import pandas as pd
import shapely.geometry
from annoy import AnnoyIndex
from geopandas import GeoDataFrame

from validateosm.compare.footprint import DescriptorFootprint
from validateosm.source.source import (
    Source,
)
from validateosm.compare.aggregate import DescriptorAggregate
from validateosm.compare.data import DescriptorData


class Compare:
    data = DescriptorData()
    aggregate = DescriptorAggregate()
    footprint = DescriptorFootprint()
    batch: Union[GeoDataFrame]
    sources: dict[str, Source]

    def __init__(self, *sources: Type[Source] | Iterable[Type[Source]], redo=False):
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
        self._redo = redo
        # TODO: Perhaps manipulate the bboxes for more efficent loading

    @functools.cached_property
    def names(self) -> list[str]:
        names =  list(self.sources.keys())
        names.sort()
        return names

    def __repr__(self):
        return f'{self.__class__.__name__}{self.names}'

    def __getitem__(self, item: str) -> Source:
        return self.sources[item]

    # def _identify(self, gdf: GeoDataFrame) -> GeoDataFrame:
    #     source: Source = next(iter(self.sources.values()))
    #     identity = source.identify(gdf)
    #     gdf = gdf.set_index(identity, drop=False, append=True)
    #     gdf = gdf.reorder_levels(['ubid', 'name']).sort_index()
    #     return gdf

    @property
    def redo(self) -> bool:
        return self._redo

    @redo.setter
    def redo(self, value):
        if not isinstance(value, bool):
            raise TypeError(type(value))
        self._redo = value

    @functools.cached_property
    def directory(self) -> Path:
        return Path(inspect.getfile(self.__class__)).parent / '_'.join(self.names)
