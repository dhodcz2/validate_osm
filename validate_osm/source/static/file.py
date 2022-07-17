import functools
from collections.abc import Collection, Container
from collections import UserList
from dataclasses import dataclass, field
import geopandas as gpd
import pandas as pd

from geopandas import GeoDataFrame, GeoSeries
from pandas import Series, DataFrame

import abc
import concurrent
import logging
import os
from pathlib import Path
from typing import Iterable, Union, Iterator, Type

import concurrent.futures
import requests

# noinspection PyUnreachableCode
if False:
    from ..source import Source


@dataclass(repr=False)
class StructFile:
    # TODO: Perhaps instantiate StructFile as StructFileFactory
    url: str = field(init=False)
    source: Type['Source'] = field(init=False)  # TODO: How can we make this automatic?
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


    @property
    def raw(self) -> Path:
        return self.source.__class__.resource.raw / self.name

    @property
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
    source: Type['Source'] = field(repr=False)
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

    @staticmethod
    def load(paths: Iterable[Path]) -> GeoDataFrame:
        # noinspection PyTypeChecker
        return pd.concat(map(StructFile.load, paths))

    def __repr__(self):
        return f'{self.source.name}.{self.name}'

    def __delete__(self, instance):
        for file in self.files:
            file.path.unlink()

class ListFiles(UserList):
    @property
    def data_(self) -> Iterator[Path]:
        for file in self.data_:
            if isinstance(file, StructFiles):
                yield from file.data
            elif isinstance(file, StructFile):
                yield file.data
            else:
                raise TypeError(f'{file} is not a StructFile or StructFiles')

    @property
    def raw(self) -> Iterator[Path]:
        for file in self.data_:
            if isinstance(file, StructFiles):
                yield from file.raw
            elif isinstance(file, StructFile):
                yield file.raw
            else:
                raise TypeError(f'{file} is not a StructFile or StructFiles')

    @staticmethod
    def load(paths: Iterable[Path]) -> GeoDataFrame:
        # noinspection PyTypeChecker
        return pd.concat(map(StructFile.load, paths))