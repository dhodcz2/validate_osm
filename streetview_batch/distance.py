import inspect

import pyproj
import math

import geopandas as gpd

from dataclasses import field, dataclass
import dataclasses
import functools

import numpy as np
import pandas as pd
from pathlib import Path
from typing import Union
from typing import Iterator, Iterable
import abc

import shapely.geometry
import skimage.io


@dataclass
class StructIterableCoords:
    geographic: Iterable[tuple[float, float]] = field(repr=False)
    metric: Iterable[tuple[float, float]] = field(repr=False)
    tile: Iterable[tuple[int, int]] = field()

    def tile_dimension(self) -> float:
        # TODO: Return the size in meters of a tile
        ...




class Distance(abc.ABC):
    @abc.abstractmethod
    def camera(self) -> StructIterableCoords:
        ...

    @abc.abstractmethod
    def heading(self) -> Iterator[float]:
        ...

    @abc.abstractmethod
    def bird(self) -> Iterator[str | Path]:
        ...

    @functools.cached_property
    def wall(self) -> StructIterableCoords:
        camera = self.camera()
        for g, m, t, h, b in zip(
                camera.geographic, camera.metric, camera.tile, self.heading(), self.bird()
        ):
            image = skimage.io.imshow(b)





class BatchDistance(Distance):
    # TODO: Each poid has a left and right component
    def __init__(self, input_: str, camera: str):
        with open(input_) as f:
            skiprows = [
                i
                for i, line in enumerate(f.readlines())
                if line.startswith('E')
            ]
        input_ = pd.read_csv(
            input_,
            skiprows=skiprows,
            usecols=[1, 2, 3],
            names=['poid', 'lat', 'lon']
        )
        camera = pd.read_csv(
            camera,
            skiprows=skiprows,
            usecols=[2],
            names=['heading']
        )
        self.resource = pd.concat((camera, input_), axis=1)

    def poid(self) -> Iterator[str]:
        for poid in self.resource['poid']:
            yield poid
            yield poid

    def heading(self) -> Iterator[float]:
        # yield from self.resource['heading']
        for heading in self.resource['heading']:
            yield (heading + 45) % 360
            heading -= 45
            if heading < 0:
                heading += 360
            yield heading

    @functools.cached_property
    def camera(self) -> StructIterableCoords:
        def geographic():
            for x, y in zip(self.resource['lon'], self.resource['lat']):
                yield x, y
                yield x, y

        def metric() -> Iterator[tuple[float, float]]:
            trans = pyproj.Transformer.from_crs('epsg:4326', 'epsg:3857', always_xy=True)
            for x, y in trans.transform(self.resource['lon'], self.resource['lat']):
                yield x, y
                yield x, y

        def tile() -> Iterator[tuple[int, int]]:
            for lon, lat in zip(self.resource['lon'], self.resource['lat']):
                lat_rad = math.radians(lat)
                n = 2.0 ** 19
                xtile = int((lon + 180.0) / 360.0 * n)
                ytile = int((1.0 - math.asinh(math.tan(lat_rad)) / math.pi) / 2.0 * n)
                yield xtile, ytile
                yield xtile, ytile

        return StructIterableCoords(list(geographic()), list(metric()), list(tile()))

    def bird(self) -> Iterator[str | Path]:
        path = Path(inspect.getfile(BatchDistance)).parent / 'static' / 'street'
        poids = iter(self.poid())
        for poid in poids:
            next(poids)
            yield path / f'{poid}_left'
            yield path / f'{poid}_right'
