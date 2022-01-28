import geopandas as gpd
import functools
import os
import shutil
import warnings

import requests
import inspect
from pathlib import Path
from typing import Union, Iterable, Optional, Type, Iterator
import zipfile
# TODO: Where should args be defined?
from ValidateOSM.source.source import Source, args

"""
Create directory under /static/Source/*
Download file
Unzip if is extension .zip
returns the Union[object, Iterator[objects]
    Can the programmer specify which specific files within the zip must be loaded?
Option to remove file from disk because these may be several gigabytes
"""


# TODO: How can we best handle downloading multiple at the same time?

class Static:
    def __init__(
            self,
            url: str,
            unzipped: Optional[Path, Iterable[Path]] = None
    ):
        self.url = url
        self.filename = url.rpartition('/')
        self.unzipped = [unzipped] if isinstance(unzipped) else list(unzipped)

    def __get__(self, instance: Source, owner: Type[Source]) \
            -> Union[gpd.GeoDataFrame, Iterator[gpd.GeoDataFrame], 'Static']:
        self._instance = instance
        self._owner = owner
        if instance is None:
            return self
        if not self.path.exists():
            warnings.warn(
                f"Downloading {self.path} for {self.__class__.__name__}; this is occurring unthreaded and therefore "
                f"stalling the rest of the process.")
            self.get()
        if not self.unzipped:
            return gpd.read_file(
                filename=self.path,
                bbox=self._instance.bbox.raw.cartesian,
                rows=100 if args.debug else None,
            )
        else:
            # TODO: Source/file.zip -> Source/file/[**/*]
            path = self.path.parent / self.path.name.rpartition('.')[0]
            if self.unzipped is None:
                files: list[Path] = [
                    p
                    for p in path.glob('**/*')
                    if p.is_file()
                ]
            else:
                files: list[Path] = [
                    path / file
                    for file in self.unzipped
                ]
                for file in files:
                    if not file.exists():
                        raise FileNotFoundError(file)
            if len(files) == 1:
                return gpd.read_file(
                    filename=files[0],
                    bbox=self._instance.bbox.raw.cartesian,
                    rows=100 if args.debug else None
                )
            else:
                return (
                    gpd.read_file(
                        filename=file,
                        bbox=self._instance.bbox.raw.cartesian,
                        rows=100 if args.debug else None
                    )
                    for file in files
                )

    @functools.cached_property
    def path(self) -> Path:
        # Download to /static/Source/*
        path = Path(inspect.getfile(self._owner)).parent / 'static' / self._owner.__name__
        if not path.exists():
            path.mkdir()
        path /= self.filename
        return path

    def get(self) -> None:
        response = requests.get(self.url, stream=True)
        response.raise_for_status()
        with open(self.path, 'wb') as file:
            for block in response.iter_content(1024):
                file.write(block)
        partition = self.path.name.rpartition('.')
        if partition[2] in {'zip', 'rar'}:
            shutil.unpack_archive(self.path)

    def delete(self) -> None:
        os.remove(self.path.parent)

    @staticmethod
    def getall(*args: list[Type]) -> None:
        # TODO: Using threading, implement concurrent downloads of all the requested static files.
        #   This allows the user to simply open up the console, call Static.getall(SourceOSM, Source, ...),
        #   go out for lunch, and come back with all the needed static files for the analytics.
        ...
