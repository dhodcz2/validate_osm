from IPython.core.display import Image, display
import PIL
import io
import abc
import functools
import inspect
import math
import os
import warnings
from dataclasses import field, dataclass
from pathlib import Path
from typing import Collection, Type
from typing import Iterator, Iterable, Optional
from zipfile import ZipFile

import PIL
import matplotlib.pyplot
import matplotlib.pyplot as plt
import numpy
import numpy as np
import pandas as pd
import pyproj
import shapely.geometry
import skimage.io
from geopandas import GeoDataFrame, GeoSeries
from pandas import DataFrame
from pandas import Series

from streetview_batch.slippy import deg2num, num2deg


class Points:
    __slots__ = ['x', 'y']

    def __repr__(self):
        tuples = (
            str(tup)
            for tup, _ in zip(self, range(3))
        )
        return ', '.join(tuples) + '...'

    def __init__(self, x: Iterable[float], y: Iterable[float]):
        self.x = x
        self.y = y

    def __iter__(self) -> Collection[tuple[float, float]]:
        yield from zip(self.x, self.y)


class Geographic(Points):
    @classmethod
    def from_metric(cls, points: 'Metric'):
        trans = pyproj.Transformer.from_crs(4326, 3857)
        y, x = trans.transform(points.y, points.x)
        return Geographic(x, y)


class Metric(Points):
    @classmethod
    def from_geographic(cls, points: 'Geographic'):
        trans = pyproj.Transformer.from_crs(4326, 3857)
        y, x = trans.transform(points.y, points.x)
        return Metric(x, y)


class Tiles(Points):
    @classmethod
    def from_geographic(cls, geographic: Geographic) -> 'Tiles':
        tiles = [
            deg2num(x, y, 19)
            for x, y in geographic
        ]
        return Tiles(
            x=[x for x, y in tiles],
            y=[y for x, y in tiles]
        )

    def to_geographic(self) -> Geographic:
        points = [
            num2deg(x, y, 19)
            for x, y in self
        ]
        return Geographic(
            x=[x for x, y in points],
            y=[y for x, y in points]
        )

    @functools.cached_property
    def nw_bound(self) -> Metric:
        geographic = self.to_geographic()
        metric = Metric.from_geographic(geographic)
        return metric

    @functools.cached_property
    def se_bound(self) -> Metric:
        tiles = Tiles(
            x=[x + 1 for x in self.x],
            y=[y + 1 for y in self.y],
        )
        geographic = tiles.to_geographic()
        return Metric.from_geographic(geographic)

    @functools.cached_property
    def displacement(self) -> Metric:
        x = [
            se[0] - nw[0]
            for nw, se in zip(self.nw_bound, self.se_bound)
        ]
        y = [
            se[1] - nw[1]
            for nw, se in zip(self.nw_bound, self.se_bound)
        ]
        return Metric(x, y)


@dataclass
class StructIterableCoords:
    geographic: Geographic = field(repr=False)
    metric: Metric = field(repr=False)
    tiles: Tiles = field(repr=False)

    @classmethod
    def from_geographic(cls, geographic: Geographic) -> 'StructIterableCoords':
        return StructIterableCoords(
            geographic,
            Metric.from_geographic(geographic),
            Tiles.from_geographic(geographic)
        )

    @classmethod
    def from_metric(cls, metric: Metric) -> 'StructIterableCoords':
        geographic = Geographic.from_metric(metric)
        return StructIterableCoords(
            geographic,
            metric,
            Tiles.from_geographic(geographic)
        )

    @classmethod
    def from_tiles(cls, tiles: Tiles) -> 'StructIterableCoords':
        geographic = tiles.to_geographic()
        metric = Metric.from_geographic(geographic)
        return StructIterableCoords(
            geographic,
            metric,
            tiles
        )

    def __repr__(self):
        return repr(self.geographic)


class Distance(abc.ABC):
    def __init__(
            self,
            cameras: StructIterableCoords | tuple,
            headings: Collection[float] | float,
            images: str | Path,
            poid: Collection[str] | str,
            **kwargs
    ):
        if isinstance(cameras, (tuple, list)):
            geographic = Geographic(x=[cameras[0]], y=[cameras[1]])
            self.camera = StructIterableCoords.from_geographic(geographic)
        elif isinstance(cameras, StructIterableCoords):
            self.camera = cameras
        else:
            raise TypeError(cameras)

        if isinstance(headings, float):
            self.heading = (headings,)
        elif isinstance(headings, Collection):
            self.heading = headings
        else:
            raise TypeError(headings)

        if isinstance(images, (str, Path)):
            self.image = images
        elif isinstance(images, Collection):
            self.image = images
        else:
            raise TypeError(images)

        if isinstance(poid, (str)):
            self.poid = (poid,)
        elif isinstance(poid, Collection):
            self.poid = poid
        else:
            raise TypeError(poid)

    @functools.cached_property
    def _northwest(self) -> StructIterableCoords:
        return StructIterableCoords.from_tiles(self.camera.tiles)

    @functools.cached_property
    def _displacement(self) -> list[tuple[float, float]]:
        def displacement() -> Iterator[tuple[float, float]]:
            for camera, tile_displacement, heading, image, northwest_bound, tile in zip(
                    self.camera.metric,
                    self.camera.tiles.displacement,
                    self.heading,
                    self.image,
                    self._northwest.metric,
                    self.camera.tiles
            ):
                theta = (450 - heading) % 360
                red = image[:, :, 0]
                slope = math.tan(math.radians(theta))
                buffer = 0
                yinc = 1 if theta < 180 else -1
                xinc = 1 if heading < 180 else -1
                slope = abs(slope)
                # TODO: Why aren't tiles aligned?

                # epsg:3857 to pixels
                cx = x = int(
                    (camera[0] - northwest_bound[0])
                    / tile_displacement[0]
                    * 255
                )
                cy = y = int(
                    (camera[1] - northwest_bound[1])
                    / tile_displacement[1]
                    * 255
                )
                if not (0 <= x <= 255) or not (0 <= y <= 255):
                    raise RuntimeError('you did this wrong')

                # This is for before I try it in Cython. We can use pythonic object assignment to compress this code,
                #   but that would cost us speed which actually matters here
                if slope > 1:
                    # buffer stores x
                    slope = 1 / slope
                    match xinc, yinc:
                        case 1, 1:
                            while x <= 255 and y <= 255:
                                if red[x, y]:
                                    break
                                elif buffer >= 1:
                                    buffer -= 1
                                    x += 1
                                else:
                                    buffer += slope
                                    y += 1
                            else:
                                yield None, None
                                continue
                        case 1, -1:
                            while x <= 255 and y >= 0:
                                if red[x, y]:
                                    break
                                elif buffer >= 1:
                                    buffer -= 1
                                    x += 1
                                else:
                                    buffer += slope
                                    y -= 1
                            else:
                                yield None, None
                                continue
                        case -1, 1:
                            while x >= 0 and y <= 255:
                                if red[x, y]:
                                    break
                                elif buffer >= 1:
                                    buffer -= 1
                                    x -= 1
                                else:
                                    buffer += slope
                                    y += 1
                            else:
                                yield None, None
                                continue
                        case -1, -1:
                            while x >= 0 and y >= 0:
                                if red[x, y]:
                                    break
                                elif buffer >= 1:
                                    buffer -= 1
                                    x -= 1
                                else:
                                    buffer += slope
                                    y -= 1
                            else:
                                yield None, None
                                continue

                else:
                    # buffer stores y
                    match xinc, yinc:
                        case 1, 1:
                            while x <= 255 and y <= 255:
                                if red[x, y]:
                                    break
                                elif buffer >= 1:
                                    buffer -= 1
                                    y += 1
                                else:
                                    buffer += slope
                                    x += 1
                            else:
                                yield None, None
                                continue
                        case 1, -1:
                            while x <= 255 and y >= 0:
                                if red[x, y]:
                                    break
                                elif buffer >= 1:
                                    buffer -= 1
                                    y -= 1
                                else:
                                    buffer += slope
                                    x += 1
                            else:
                                yield None, None
                                continue
                        case -1, 1:
                            while x >= 0 and y <= 255:
                                if red[x, y]:
                                    break
                                elif buffer >= 1:
                                    buffer -= 1
                                    y -= 1
                                else:
                                    buffer = + slope
                                    x -= 1
                            else:
                                yield None, None
                                continue
                        case -1, -1:
                            while x >= 0 and y >= 0:
                                if red[x, y]:
                                    break
                                elif buffer >= 1:
                                    buffer -= 1
                                    y -= 1
                                else:
                                    buffer = + slope
                                    x -= 1
                            else:
                                yield None, None
                                continue

                # pixels to displacement
                x = (x / 255 * tile_displacement[0])
                y = (y / 255 * tile_displacement[1])
                yield x, y

        return list(displacement())

    @functools.cached_property
    def _distance(self) -> list[Optional[float]]:
        return [
            math.sqrt(x ** 2 + y ** 2)
            if x is not None
            else None
            for x, y in self._displacement
        ]

    @functools.cached_property
    def _wall(self) -> StructIterableCoords:
        displacement = self._displacement
        metric = self.camera.metric
        x = [
            m[0] + d[0]
            if d is not None
            else None
            for d, m in zip(displacement, metric)
        ]
        y = [
            m[1] + d[1]
            if d is not None
            else None
            for d, m in zip(displacement, metric)
        ]
        metric = Metric(x, y)
        return StructIterableCoords(
            geographic=Geographic.from_metric(metric),
            metric=metric,
            tiles=self.camera.tiles
        )

    @functools.cached_property
    def distance(self) -> GeoDataFrame:
        poid = Series(self.poid, dtype='string')
        # Just for human friendly sub-indexing
        heading = Series((
            int(heading)
            for heading in self.heading
        ), dtype='int8')
        wall = GeoSeries((
            shapely.geometry.Point(g[0], g[1])
            if g is not None
            else None
            for g in self._wall.geographic
        ))
        camera = GeoSeries((
            shapely.geometry.Point(g[0], g[1])
            if g is not None
            else None
            for g in self.camera.geographic
        ))
        distance = Series(self._distance, dtype='Float64')

        return GeoDataFrame({
            'poid': poid,
            'heading': heading,
            'camera': camera,
            'wall': wall,
            'distance': distance,
        }, index=['poid', 'heading'])

    @functools.cached_property
    def plots(self) -> DataFrame:
        nw = self._northwest.metric
        camera = self.camera.metric
        wall = self._wall.metric
        displacement = self.camera.tiles.displacement
        index = self.poid
        image = self.image
        distance = self.distance

        cx = [
            int(
                (cx - nwx) / d[0] * 255
            )
            for nwx, cx, d in zip(nw.x, camera.x, displacement)
        ]
        cy = [
            int(
                (cy - ny) / d[1] * 255
            )
            for ny, cy, d in zip(nw.y, camera.y, displacement)
        ]
        wx = [
            int(
                (wx - nwx) / d[0] * 255
            )
            if wx is not None
            else None
            for nwx, wx, d in zip(nw.x, wall.x, displacement)
        ]
        wy = [
            int(
                (wy - ny) / d[1] * 255
            )
            if wy is not None
            else None
            for ny, wy, d in zip(nw.y, wall.y, displacement)
        ]
        return DataFrame({
            'cx': cx,
            'cy': cy,
            'wx': wx,
            'wy': wy,
            'image': image,
            'distance': distance
        }, index=index)

    def plot(self, poid: str):
        fig: matplotlib.pyplot.Figure
        ax: matplotlib.pyplot.Axes
        fig, ax = plt.subplots(1, 1)
        cx, cy, wx, wy, image, distance = self.plots.loc[poid].values
        ax.imshow(skimage.io.imread(image))
        ax.set_title(f'{distance=}')
        ax.plot(cy, cx, 'wo')
        ax.plot(wy, wx, 'wo')

    def load_distance(self, path: Path | str):
        path = Path(path)
        if not path.parent.exists():
            os.makedirs(path)
        method = f"to_{path.name.rpartition('.')[2]}"
        getattr(self.distance, method)(path)

    def load_plots(self, directory: Path | str):
        path = Path(directory)
        if path.is_file():
            raise ValueError(path)
        if not path.exists():
            os.makedirs(path)
        plots = self.plots
        # TODO: Save each image to a file


class BatchDistance(Distance):
    """
    Caveat: Every streetview has one taken from left side, and one taken from right side.
        For each camera
            in each iterator
            yield twice
    """

    def __init__(self, path_inputs: str, path_metadata: str, path_images: str, **kwargs):
        """

        :param path_inputs:
            csv
            lat,lon,heading
        :param path_metadata:
            csv
            status,poid,lat,lon
        :param path_images:
            zip, directory
        """

        with open(path_inputs) as f:
            rows = np.fromiter((
                line[0]
                for line in f.readlines()
            ), dtype='U1')
        skiprows = np.fromiter((
            i
            for i, char in enumerate(rows)
            if char == 'E'
        ), dtype=int)
        line = pd.Series((
            i
            for i, char in enumerate(rows)
            if char != 'E'
        ))
        path_inputs = pd.read_csv(
            path_inputs,
            skiprows=skiprows,
            usecols=[1, 2, 3],
            names=['poid', 'lat', 'lon'],
            dtype={
                'poid': 'string',
                'lat': 'Float64',
                'lon': 'Float64',
            },
        )
        path_metadata = pd.read_csv(
            path_metadata,
            skiprows=skiprows,
            usecols=[2],
            names=['heading'],
            dtype={
                'heading': 'Float64'

            },
        )
        resource = pd.concat((path_metadata, path_inputs), axis=1)
        resource['line'] = pd.Series(iter(line), dtype='int16')
        resource = resource.set_index('poid', drop=False)
        loc = resource['heading'].isna()
        if any(loc):
            warnings.warn('some headers in the file have heading=nan')
            resource = resource[~loc]

        def poid() -> Iterator[str]:
            for poid in resource['poid']:
                yield poid
                yield poid

        def heading() -> Iterator[float]:
            for heading in resource['heading']:
                yield (heading + 45) % 360
                heading -= 45
                if heading < 0:
                    heading += 360
                yield heading

        def camera() -> StructIterableCoords:
            def lon():
                for x in resource['lon']:
                    yield x
                    yield x

            def lat():
                for y in resource['lat']:
                    yield y
                    yield y

            geographic = Geographic(x=list(lon()), y=list(lat()))
            metric = Metric.from_geographic(geographic)
            tiles = Tiles.from_geographic(geographic)
            return StructIterableCoords(geographic, metric, tiles)

        super(BatchDistance, self).__init__(
            cameras=camera(),
            headings=list(heading()),
            images=path_images,
            poid=list(poid()),
            **kwargs
        )

    class DescriptorImages:
        directory: str | Path

        def __get__(self, instance: 'BatchDistance', owner: Type['BatchDistance']):
            self.instance = instance
            self.owner = owner
            return self

        def __set__(self, instance, value):
            self.instance = instance
            self.directory = Path(value)

        def __getitem__(self, tile: tuple[float, float]):
            x, y = tile
            directory = self.directory
            if directory.is_dir():
                path = directory / f"19_{y}_{x}_pred.png"
                image = skimage.io.imread(path)
            elif directory.name.endswith('.zip'):
                with ZipFile(directory) as zf:
                    path = f'best_images/19_{y}_{x}_pred.png'
                    with zf.open(path) as f:
                        image = skimage.io.imread(f)
            return image

        def __iter__(self):
            directory = self.directory
            if directory.is_dir():
                for x, y in self.instance.camera.tiles:
                    path = directory / f"19_{y}_{x}_pred.png"
                    image = skimage.io.imread(path)
                    yield image
                    yield image
            elif directory.name.endswith('.zip'):
                with ZipFile(directory) as zf:
                    for x, y in self.instance.camera.tiles:
                        directory = f'best_images/19_{y}_{x}_pred.png'
                        with zf.open(directory) as f:
                            image = skimage.io.imread(f)
                            yield image
                            yield image
            else:
                raise ValueError(directory)

    image = DescriptorImages()


if __name__ == '__main__':
    batch = BatchDistance(
        path_inputs='/home/arstneio/PycharmProjects/ValidateOSM/streetview_batch/static/camera/manhattan_gsv.csv',
        path_metadata='/home/arstneio/PycharmProjects/ValidateOSM/streetview_batch/static/input/manhattan.csv',
        path_images='/home/arstneio/PycharmProjects/ValidateOSM/streetview_batch/static/bird.zip'
    )
    batch.distance
    batch.plots
    print()
    # distance = batch.distance
    # print()
    # plots = batch.plots
