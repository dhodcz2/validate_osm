import functools
import inspect
from pathlib import Path
from typing import Union, Iterable, Type, Iterator

from geopandas import GeoDataFrame

from validateosm.compare.aggregate import DescriptorAggregate
from validateosm.compare.data import DescriptorData
from validateosm.source.footprint import CallableFootprint
from validateosm.source.source import (
    Source, BBox
)
from validateosm.compare.plot import DescriptorPlot


class Compare:
    data = DescriptorData()
    aggregate = DescriptorAggregate()
    plot = DescriptorPlot()
    batch: Union[GeoDataFrame]
    sources: dict[str, Source]

    def __init__(self, *sources: Type[Source] | Iterable[Type[Source]], ignore_file=False, bbox: BBox = None):
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
                source.ignore_file = ignore_file
        self.bbox = bbox

        self._footprint = None

    @property
    def footprint(self) -> CallableFootprint:
        if self._footprint is None:
            raise RuntimeError(f"Compare.footprint is dependent upon Compare.data; first, instantiate Compare.data")
        return self._footprint

    @footprint.setter
    def footprint(self, value):
        self._footprint = value

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
