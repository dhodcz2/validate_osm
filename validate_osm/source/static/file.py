import concurrent.futures
from functools import cached_property
from collections import UserList
from collections import UserList
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, Iterator, Callable

import geopandas as gpd
import pandas as pd
from geopandas import GeoDataFrame

# noinspection PyUnreachableCode
if False:
    from ..source import Source


@dataclass(repr=False)
class StructFile:
    url: str = field(repr=False)
    source: 'Source' = field(repr=False)
    name: str = field(repr=False, default=None)
    size: int = field(init=False, repr=False, compare=True)

    def __repr__(self):
        return f'{self.source.name}.{self.name}'

    def __post_init__(self):
        if self.name is None:
            self.name = (
                self.url.rpartition('/')[2]
                    .rpartition('.')[0]
            )
        self.size = self.raw.stat().st_size

    @cached_property
    def directory(self) -> Path:
        return self.source.__class__.resource.directory

    @cached_property
    def raw(self) -> Path:
        return self.source.__class__.resource.raw / self.name

    @cached_property
    def data(self) -> Path:
        return self.source.__class__.resource.data / self.name

    @staticmethod
    def load(path: Path) -> GeoDataFrame:
        match path.name.rpartition('.')[2]:
            case 'feather':
                return gpd.read_feather(path)
            case 'parquet':
                return gpd.read_parquet(path)
            case _:
                return gpd.read_file(path)

    def delete(self):
        self.path.unlink()


@dataclass(repr=False)
class StructFiles:
    files: list[StructFile] = field(repr=False)
    source: 'Source' = field(repr=False)
    name: str = field(repr=False, default=None)
    size: int = field(init=False, repr=False, compare=True)

    def __post_init__(self):
        self.size = sum(file.size for file in self.files)

    @property
    def data(self) -> Iterator[Path]:
        return (file.data for file in self.files)

    @property
    def raw(self) -> Iterator[Path]:
        return (file.raw for file in self.files)

    # noinspection PyTypeChecker
    @staticmethod
    def load(paths: Iterable[Path]) -> GeoDataFrame:
        def gdfs() -> Iterator[GeoDataFrame]:
            with concurrent.futures.ThreadPoolExecutor() as threads:
                yield from concurrent.futures.as_completed([
                    threads.submit(StructFile.load, path)
                    for path in paths
                ])

        return pd.concat(gdfs())

    def __repr__(self):
        return f'{self.source.name}.{self.name}'

    def __delete__(self, instance):
        for file in self.files:
            file.path.unlink()


class ListFiles(UserList):
    data: list[StructFile, StructFiles]
    __iter__: Callable[[], Iterator[StructFile | StructFiles]]
    __getitem__: Callable[[int], StructFile | StructFiles]

    @property
    def url(self) -> Iterator[str]:
        for file in self.data:
            if isinstance(file, StructFiles):
                for f in file.files:
                    yield f.url
            elif isinstance(file, StructFile):
                yield file.url
            else:
                raise TypeError(f'{file} is not a StructFile or StructFiles')

    @property
    def data_(self) -> Iterator[Path]:
        for file in self.data_:
            if isinstance(file, StructFiles):
                for f in file.files:
                    yield f.data
            elif isinstance(file, StructFile):
                yield file.data
            else:
                raise TypeError(f'{file} is not a StructFile or StructFiles')

    @property
    def raw(self) -> Iterator[Path]:
        for file in self.data_:
            if isinstance(file, StructFiles):
                for f in file.files:
                    yield f.raw
            elif isinstance(file, StructFile):
                yield file.raw
            else:
                raise TypeError(f'{file} is not a StructFile or StructFiles')

    @staticmethod
    def load(paths: Iterable[Path]) -> GeoDataFrame:
        # noinspection PyTypeChecker
        return pd.concat(map(StructFile.load, paths))
