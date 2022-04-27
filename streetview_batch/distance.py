import abc
import functools
import itertools
import math
import os
import warnings
from dataclasses import field, dataclass
from functools import cached_property
from pathlib import Path
from typing import Collection, Type, Any
from typing import Iterator, Iterable
from zipfile import ZipFile

import geopy.distance
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pyproj
import skimage.io
from geopandas import GeoDataFrame, GeoSeries
from pandas import DataFrame
from pandas import Series

# TODO: Is Projected entirely pointless?
from streetview_batch.slippy import deg2num, num2deg


# import pyximport
# pyximport.install()
# from streetview_batch.cdistance import cdistance


class Points(DataFrame):
    x: Series
    y: Series
    crs: Any

    def __init__(self, x: Iterable[float], y: Iterable[float], crs=None, **kwargs):
        super(Points, self).__init__({'x': x, 'y': y}, **kwargs)
        super(Points, self).__setattr__('crs', crs)

    def __iter__(self) -> Iterator[tuple[float]]:
        yield from self.values

    @property
    def geoseries(self):
        loc = self.x.notna()
        return GeoSeries.from_xy(
            self.x[loc],
            self.y[loc],
            crs=self.crs,
            index=self.index[loc]
        )


class Geographic(Points):
    def __init__(self, x: Iterable[float], y: Iterable[float], crs=4326, **kwargs):
        super(Geographic, self).__init__(x, y, crs=crs, **kwargs)

    @classmethod
    def from_projected(cls, points: 'Projected'):
        trans = pyproj.Transformer.from_crs(points.crs, 4326, always_xy=True)
        x, y = trans.transform(points.x, points.y)
        return Geographic(x, y, crs=4326, index=points.index)


class Projected(Points):
    def __init__(self, x: Iterable[float], y: Iterable[float], crs=3857, **kwargs):
        super(Projected, self).__init__(x, y, crs, **kwargs)

    @classmethod
    def from_geographic(cls, points: 'Geographic'):
        trans = pyproj.Transformer.from_crs(points.crs, 3857, always_xy=True)
        x, y = trans.transform(points.x, points.y)
        return Projected(x, y, crs=3857, index=points.index)


class Tiles(Points):
    @classmethod
    def from_geographic(cls, geographic: Geographic) -> 'Tiles':
        tiles = [
            deg2num(x, y, 19)
            for x, y in geographic
        ]
        x = [x for x, y in tiles]
        y = [y for x, y in tiles]
        return Tiles(x, y, index=geographic.index)

    def to_geographic(self) -> Geographic:
        points = [
            num2deg(x, y, 19)
            for x, y in self
        ]
        return Geographic(
            x=[x for x, y in points],
            y=[y for x, y in points],
            crs=4326,
            index=self.index
        )

    @cached_property
    def nw_bound(self) -> Projected:
        geographic = self.to_geographic()
        projected = Projected.from_geographic(geographic)
        return projected

    @cached_property
    def se_bound(self) -> Projected:
        tiles = Tiles(
            x=self.x + 1,
            y=self.y + 1,
        )
        geographic = tiles.to_geographic()
        return Projected.from_geographic(geographic)

    @cached_property
    def displacement(self) -> Projected:
        return Projected(
            x=self.se_bound.x - self.nw_bound.x,
            y=self.se_bound.y - self.nw_bound.y,
        )


@dataclass
class StructIterableCoords:
    geographic: Geographic = field(repr=False)
    projected: Projected = field(repr=False)
    tiles: Tiles = field(repr=False)

    @classmethod
    def from_geographic(cls, geographic: Geographic) -> 'StructIterableCoords':
        return StructIterableCoords(
            geographic,
            Projected.from_geographic(geographic),
            Tiles.from_geographic(geographic)
        )

    @classmethod
    def from_projected(cls, projected: Projected) -> 'StructIterableCoords':
        geographic = Geographic.from_projected(projected)
        return StructIterableCoords(
            geographic,
            projected,
            Tiles.from_geographic(geographic)
        )

    @classmethod
    def from_tiles(cls, tiles: Tiles) -> 'StructIterableCoords':
        geographic = tiles.to_geographic()
        projected = Projected.from_geographic(geographic)
        return StructIterableCoords(
            geographic,
            projected,
            tiles
        )

    def __repr__(self):
        return repr(self.geographic)


class Distance(abc.ABC):
    _camera: StructIterableCoords
    _heading: Collection[float]
    _image: Iterable[np.array]
    _poid: Iterable[str]
    _index: pd.MultiIndex
    _projection: Any

    @cached_property
    def _northwest(self) -> StructIterableCoords:
        return StructIterableCoords.from_tiles(self._camera.tiles)

    @cached_property
    def _displacement(self) -> DataFrame:
        # TODO: implement displacement in Cython
        def displacement() -> Iterator[tuple[float, float]]:
            for camera, tile_displacement, heading, image, northwest, tile, poid in zip(
                    self._camera.projected,
                    self._camera.tiles.displacement,
                    self._heading,
                    self._image,
                    self._northwest.projected,
                    self._camera.tiles,
                    self._poid
            ):
                xlen = tile_displacement[0]
                ylen = tile_displacement[1]
                theta = (450 - heading) % 360
                red = image[:, :, 0]
                slope = math.tan(math.radians(theta))
                buffer = 0
                slope = abs(slope)

                ccam = cwall = int(
                    (camera[0] - northwest[0]) / xlen * 255
                )
                rcam = rwall = int(
                    (camera[1] - northwest[1]) / ylen * 255
                )
                # 1 is right, -1 is left
                cinc = 1 if heading < 180 else -1
                # -1 is up, 1 is down
                rinc = -1 if theta < 180 else 1

                if slope > 1:
                    # r >> c
                    slope = 1 / slope
                    while 0 <= cwall <= 255 and 0 <= rwall <= 255:
                        if red[rwall, cwall]:
                            y = (rwall - rcam) / 255 * ylen
                            x = (cwall - ccam) / 255 * xlen
                            yield x, y
                            break
                        elif buffer >= 1:
                            buffer += -1
                            cwall += cinc
                        else:
                            buffer += slope
                            rwall += rinc
                    else:
                        yield np.nan, np.nan
                else:
                    # c >> r
                    while 0 <= cwall <= 255 and 0 <= rwall <= 255:
                        if red[rwall, cwall]:
                            y = (rwall - rcam) / 255 * ylen
                            x = (cwall - ccam) / 255 * xlen
                            yield x, y
                            break
                        elif buffer >= 1:
                            buffer += -1
                            rwall += rinc
                        else:
                            buffer += slope
                            cwall += cinc
                    else:
                        yield np.nan, np.nan

        return DataFrame(displacement(), columns=['x', 'y'], index=self._index)

    # TODO: small presentation on oak ridge building data

    @property
    def _distance(self) -> pd.Series:
        # wall has na
        wall = self._wall
        camera = self._camera.geographic

        wall = wall.geoseries.to_crs(4326)
        # camera = camera.geoseries.to_crs(4326)

        index = wall.index.intersection(camera.index)
        wall = wall.loc[index]
        camera = camera.loc[index]

        wall_tuples = zip(wall.y, wall.x)
        camera_tuples = zip(camera.y, camera.x)
        distance = Series((
            geopy.distance.distance(w, c).meters
            for w, c in zip(wall_tuples, camera_tuples)
        ), dtype='Float64', index=index)

        return distance

        # loc = wall.x.notna()
        # index = self._index[loc]
        # camera: DataFrame = camera.loc[loc]
        # wall: DataFrame = wall.loc[loc]
        #
        #
        # y, x = trans.transform(wall.y, wall.x)
        # wall_tuples = zip(y, x)
        # camera_tuples = zip(camera.y, camera.x)
        #
        # distance = Series((
        #     geopy.distance.distance(w, c).meters
        #     for w, c in zip(wall_tuples, camera_tuples)
        # ), dtype='Float64', index=index)
        # return distance

    @cached_property
    def _wall(self) -> Projected:
        displacement = self._displacement
        return Projected(
            x=displacement['x'] + self._camera.projected.x,
            y=displacement['y'] + self._camera.projected.y,
            crs=3857,
            index=self._index
        )

    @cached_property
    def distance(self) -> GeoDataFrame:
        tile = pd.Series(iter(self._camera.tiles), index=self._index)
        return GeoDataFrame({
            'distance': self._distance,
            'camera': self._camera.geographic.geoseries,
            'wall': self._wall.geoseries,
            'tile': tile,
        })

    @functools.cached_property
    def plots(self) -> DataFrame:
        xlens = self._camera.tiles.displacement.x
        ylens = self._camera.tiles.displacement.y
        displacement = self._displacement
        northwest = self._northwest.projected
        distance = self.distance
        camera = self._camera.projected

        colcams: Series = camera.x - northwest.x
        colcams /= xlens * 255
        colcams = colcams.astype('uint16')

        rowcams: Series = camera.y - northwest.y
        rowcams /= ylens * 255
        rowcams = rowcams.astype('uint16')

        colwalls: Series = displacement['x'] / xlens * 255
        colwalls = colwalls.astype('uint16')
        colwalls += colcams

        rowwalls: Series = displacement['y'] / ylens * 255
        rowwalls = rowwalls.astype('uint16')
        rowwalls += rowcams

        distance = distance.assign(
            colcam=colcams,
            rowcam=rowcams,
            colwall=colwalls,
            rowwall=rowwalls,
        )
        return distance

    def plot(self, poid: str, heading: int):
        fig, ax = plt.subplots()
        series = self.plots.loc[poid, heading]
        image = self._image[series['tile']]
        ax.imshow(image)
        ax.plot(*series[['colcam', 'rowcam']], 'wo')
        ax.plot(*series[['colwall', 'rowwall']], 'wo')
        distance = series['distance']
        if pd.notna(distance):
            ax.set_title(f'{distance=:.1f}')
        camera = series['camera']
        print(f'Camera: {camera.y, camera.x}')
        wall = series['wall']
        if wall is not None:
            print(f'Wall: {wall.y, wall.x}')

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


class DescriptorImages:
    def __get__(self, instance: 'BatchDistance', owner: Type['BatchDistance']):
        self.instance = instance
        self.owner = owner
        return self

    def __getitem__(self, item: tuple[float, float]):
        x, y = item
        path_images = self.instance.path_images
        if path_images.is_dir():
            path = path_images / f"19_{y}_{x}_pred.png"
            image = skimage.io.imread(path)
        elif path_images.name.endswith('.zip'):
            with ZipFile(path_images) as zf:
                path = f'best_images/19_{y}_{x}_pred.png'
                with zf.open(path) as f:
                    image = skimage.io.imread(f)
        return image

    def __iter__(self):
        path_images = self.instance.path_images
        tiles = self.instance._camera.tiles
        if path_images.is_dir():
            for x, y in tiles:
                path = path_images / f'19_{y}_{x}_pred.png'
                yield skimage.io.imread(path)
        elif path_images.name.endswith('.zip'):
            with ZipFile(path_images) as zf:
                for x, y in tiles:
                    directory = f'best_images/19_{y}_{x}_pred.png'
                    with zf.open(directory) as f:
                        yield skimage.io.imread(f)
        else:
            raise ValueError(path_images)


class BatchDistance(Distance):
    _image = DescriptorImages()

    def __init__(self, path_inputs: str, path_metadata: str, path_images: str):
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
        self.path_images = Path(path_images)

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
        inputs = pd.read_csv(
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
        metadata = pd.read_csv(
            path_metadata,
            skiprows=skiprows,
            usecols=[2],
            names=['heading'],
            dtype={
                'heading': 'Float64'

            },
        )
        resource = pd.concat((metadata, inputs), axis=1)
        resource['line'] = pd.Series(iter(line), dtype='int16')
        # resource = resource.set_index('poid', drop=False)
        loc = resource['heading'].isna()
        if any(loc):
            warnings.warn('some headers in the file have heading=nan')
            resource = resource[~loc]

        def headings():
            for heading in resource['heading']:
                yield (heading + 90) % 360
                heading += -90
                if heading < 0:
                    heading += 360
                yield heading

        source = DataFrame({
            'poid': resource['poid'].repeat(2).reset_index(drop=True),
            'heading': Series(headings(), dtype='Float64'),
            'lon': resource['lon'].repeat(2).reset_index(drop=True),
            'lat': resource['lat'].repeat(2).reset_index(drop=True),
        })
        # It is possible for the resource to have implicated duplicate headings at a particular location
        index = pd.MultiIndex.from_tuples(zip(
            source['poid'],
            source['heading'].round(0).astype('uint16')
        ), names=['poid', 'heading'])
        loc = ~index.duplicated()
        source = source.loc[loc]
        index = index[loc]
        source.index = index

        self._index = index
        self._source = source
        self._poid = source['poid']
        self._heading = source['heading']

        geographic = Geographic(x=self._source['lon'], y=self._source['lat'])
        projected = Projected.from_geographic(geographic)
        tiles = Tiles.from_geographic(geographic)
        self._camera = StructIterableCoords(geographic, projected, tiles)


if __name__ == '__main__':
    batch = BatchDistance(
        path_inputs='/home/arstneio/PycharmProjects/ValidateOSM/streetview_batch/static/camera/manhattan_gsv.csv',
        path_metadata='/home/arstneio/PycharmProjects/ValidateOSM/streetview_batch/static/input/manhattan.csv',
        path_images='/home/arstneio/PycharmProjects/ValidateOSM/streetview_batch/static/bird.zip'
    )

    batch.distance
    print()
