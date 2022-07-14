# TODO: For every footprint, the head should contain the centroid,
#   and there should be a minimum thickness of 2 tiles at the centroid
# TODO: Multipolygons are currently generating tiles between polygons. Is this appropriate?
import os
import tempfile
import posixpath
import zipfile

import pandas as pd
import pygeos.creation

sep = posixpath.sep

from pandas import IndexSlice as idx
from typing import Union, Optional
import folium
import functools
import geopandas as gpd
from geopandas import GeoDataFrame
from pandas import IndexSlice as idx
# import pluscodes.util as util
try:
    import pluscodes.util as util
except ImportError:
    import util

import numpy as np
from geopandas import GeoSeries

posixpath.sep = sep  # somehow, the geographic dependencies are deleting posixpath.sep

__all__ = ['PlusCodes']

# TODO: lookup


class DescriptorLoc:
    def __get__(self, instance: 'PlusCodes', owner):
        self.pluscodes = instance
        return self

    def __getitem__(self, item: np.ndarray) -> 'PlusCodes':
        pc = self.pluscodes
        if np.issubdtype(item.dtype, np.integer):
            footprints = pc.footprints.loc[item]
            heads = pc.heads.loc[idx[item, :]]
            tiles = pc.tiles.loc[idx[item, :, :]]

        elif np.issubdtype(item.dtype, np.string_):
            heads = pc.heads.loc[idx[:, item]]
            footprints = heads.index.get_level_values(0)
            tiles = pc.tiles.loc[idx[footprints, :, :]]
            footprints = pc.footprints.loc[idx[footprints]]

        elif np.issubdtype(item.dtype, np.bool_):
            heads = pc.heads.loc[item]
            footprints = pc.footprints.loc[item]
            tiles = pc.tiles.loc[idx[footprints.index, :, :]]

        else:
            raise TypeError(f'{item.dtype} is not supported')
        return PlusCodes(heads, footprints, tiles)


#
class DescriptorIloc:
    def __get__(self, instance: 'PlusCodes', owner):
        self.pluscodes = instance
        return self

    def __getitem__(self, item) -> 'PlusCodes':
        pc = self.pluscodes
        heads = pc.heads.iloc[item]
        footprints = pc.footprints.iloc[item]
        tiles = pc.tiles.loc[idx[footprints.index, :, :]]
        return PlusCodes(heads, footprints, tiles)


class DescriptorCx:
    def __get__(self, instance: 'PlusCodes', owner):
        self.pluscodes = instance
        return self

    def __getitem__(self, item) -> 'PlusCodes':
        pc = self.pluscodes
        footprints = pc.footprints.cx[item]
        # heads = pc.heads.loc[idx[footprints.index, :]]
        tiles = pc.tiles.loc[idx[footprints.index, :, :]]
        return PlusCodes(
            footprints=footprints,
            # heads=heads,
            heads=None,
            tiles=tiles
        )
        # return PlusCodes(heads, footprints, tiles)

    def latlon(self, miny, minx, maxy, maxx) -> 'PlusCodes':
        item = (
            slice(minx, maxx),
            slice(miny, maxy)
        )
        return self.__getitem__(item)


class PlusCodes:
    # TODO: tiles and z
    loc = DescriptorLoc()
    iloc = DescriptorIloc()
    cx = DescriptorCx()

    def __init__(self, footprints: GeoDataFrame, heads: GeoDataFrame, tiles: GeoDataFrame):
        self.footprints = footprints
        self.heads = heads
        self.tiles = tiles

    def __len__(self):
        return len(self.footprints)

    @functools.cached_property
    def _total_bounds(self):
        return self.footprints.total_bounds

    def __repr__(self):
        bounds = ', '.join(
            str(val)
            for val in self.footprints.total_bounds.round(2)
        )
        return f'{self.__class__.__qualname__}[{bounds}]'

    @classmethod
    def __footprints(cls, gdf: GeoDataFrame) -> GeoDataFrame:
        footprints = gdf.to_crs(epsg=4326)
        # lengths = util.get_lengths(*footprints.geometry.bounds.values.T)
        # footprints = footprints[lengths <= 12]
        footprints = footprints.reset_index(drop=True)
        footprints.index.name = 'footprint'
        return footprints

    @classmethod
    def __heads(cls, footprints: GeoDataFrame, lengths) -> GeoDataFrame:
        # TODO: Get centroids without creating the shapely objects
        # points = footprints.representative_point()
        points = footprints.geometry.centroid
        x = points.x.values
        y = points.y.values
        bounds = util.get_bounds(x, y, lengths)
        names = util.get_strings(x, y, lengths)
        index = pd.MultiIndex.from_arrays((
            np.arange(len(footprints)),
            names
        ), names=('footprint', 'head'))
        geometry = pygeos.creation.box(bounds[:, 0], bounds[:, 1], bounds[:, 2], bounds[:, 3])
        heads = GeoDataFrame(
            index=index,
            geometry=geometry,
            crs=4326
        )
        return heads

    @classmethod
    def from_gdf(cls, gdf: Union[GeoDataFrame, GeoSeries, str]) -> 'PlusCodes':
        if isinstance(gdf, str):
            extension = gdf.rpartition('.')[-1]
            if extension == 'feather':
                gdf = gpd.read_feather(gdf)
            elif extension == 'parquet':
                gdf = gpd.read_parquet(gdf)
            else:
                gdf = gpd.read_file(gdf)

        footprints = cls.__footprints(gdf)
        bounds = footprints._bounds.T.values
        lengths = util.get_lengths(footprints, bounds)
        tiles = util.get_geoseries_tiles(footprints, lengths)
        heads = cls.__heads(footprints, lengths)
        return cls(footprints=footprints, heads=heads, tiles=tiles)

    @classmethod
    def from_file(cls, filepath: str) -> 'PlusCodes':
        with zipfile.ZipFile(filepath) as zf:
            tempdir = tempfile.gettempdir()
            footprints = zf.extract('footprints.feather', tempdir)
            tiles = zf.extract('tiles.feather', tempdir)
            heads = zf.extract('heads.feather', tempdir)
        footprints = gpd.read_feather(footprints)
        tiles = gpd.read_feather(tiles)
        heads = gpd.read_feather(heads)
        return cls(footprints=footprints, heads=heads, tiles=tiles)

    def to_file(self, path: Optional[str] = None) -> str:
        if path is None:
            tempdir = tempfile.gettempdir()
            path = os.path.join(os.getcwd(), 'pluscodes.zip')
        footprints = os.path.join(tempdir, 'footprints.feather')
        heads = os.path.join(tempdir, 'heads.feather')
        tiles = os.path.join(tempdir, 'tiles.feather')
        self.footprints.to_feather(footprints)
        self.heads.to_feather(heads)
        self.tiles.to_feather(tiles)
        with zipfile.ZipFile(path, 'w') as zip:
            zip.write(footprints)
            zip.write(heads)
            zip.write(tiles)
        return path

    def xs(self, key: Union[int, str], level) -> 'PlusCodes':
        if level == 'footprint':
            footprints = self.footprints.xs(key)
            heads = self.heads.loc[idx[footprints.index, :]]
            tiles = self.tiles.loc[idx[footprints.index, :, :]]
        elif level == 'head':
            heads = self.heads.xs(key, level='head')
            footprint = heads.index.get_level_values(0)
            tiles = self.tiles.loc[idx[footprint, :, :]]
            footprints = self.footprints.loc[idx[footprint]]
        elif level == 'tile':
            tiles = self.tiles.xs(key, level='tile')
            footprint = tiles.index.get_level_values(0)
            heads = self.heads.loc[idx[footprint.index, :]]
            footprints = self.footprints.loc[idx[footprint]]
        else:
            raise ValueError(f'{level} is not supported')
        return PlusCodes(footprints, heads, tiles)

    def explore(self, **kwargs) -> folium.Map:
        centroid = self.footprints.geometry.iloc[0].centroid
        footprints: GeoDataFrame = self.footprints
        footprints: GeoDataFrame = GeoDataFrame({
        }, geometry=footprints.geometry, crs=4326, index=footprints.index)

        heads: GeoDataFrame = self.heads
        heads: GeoDataFrame = GeoDataFrame({
            'head': heads.index.get_level_values('head'),
        }, geometry=heads.geometry, crs=4326, index=heads.index)

        tiles: GeoDataFrame = self.tiles
        tiles: GeoDataFrame = GeoDataFrame({
            'tile': tiles.index.get_level_values('tile'),
        }, geometry=tiles.geometry, crs=4326, index=tiles.index)

        head = set(heads.index.get_level_values('head'))
        tiles = tiles[~tiles.index.get_level_values('tile').isin(head)]

        m = folium.Map(
            location=(centroid.y, centroid.x),
            zoom_start=16,
        )
        footprints.explore(
            m=m,
            color='black',
            style_kwds=dict(
                fill=False,
            ),
            **kwargs,
        )
        # if heads is not None:
        heads.explore(
            m=m,
            color='blue',
            **kwargs,
        )
        tiles.explore(
            m=m,
            color='red',
            **kwargs,
        )
        return m




if __name__ == '__main__':
    import numpy as np
    import geopandas as gpd
    #
    # gdf = gpd.read_feather('/home/arstneio/Downloads/gdf.feather')
    # ne = gdf.cx[
    #      -87.62779796578965: -87.61138284607217,
    #      41.88077890032266:41.88806354070675,
    #      ]
    # pc = PlusCodes.from_gdf(ne)
    # pc.explore()
    import util.tile
    import geopandas as gpd

    gdf = gpd.read_feather('/home/arstneio/Downloads/ne.feather')
    pc = PlusCodes.from_gdf(gdf)
    pc.explore()
