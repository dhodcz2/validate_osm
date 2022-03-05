import functools
import inspect
from pathlib import Path
from typing import Union, Iterable, Type, Iterator

from geopandas import GeoDataFrame

from validateosm.compare.aggregate import DescriptorAggregate
from validateosm.compare.data import DescriptorData
from validateosm.source.footprint import Footprint
from validateosm.source.source import (
    Source,
)


class Compare:
    data = DescriptorData()
    aggregate = DescriptorAggregate()
    batch: Union[GeoDataFrame]
    sources: dict[str, Source]

    def __init__(self, *sources: Type[Source] | Iterable[Type[Source]], ignore_file=False):
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

        # Resolve footprint from sources
        footprints: Iterator[tuple[Type[Footprint], Source]] = zip(
            (source.footprint for source in sources),
            sources
        )
        footprint, source = next(footprints)
        for other_footprint, other_source in footprints:
            if other_footprint is not footprint:
                raise ValueError(f"{source.__class__.__name__}.footprint!={other_source.__class__.name}.footprint")
        self.footprint = footprint(self)


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
