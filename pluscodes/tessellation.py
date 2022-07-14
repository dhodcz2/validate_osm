import abc
# TODO: For every footprint, the head should contain the centroid,
#   and there should be a minimum thickness of 2 tiles at the centroid
# TODO: Multipolygons are currently generating tiles between polygons. Is this appropriate?
import itertools
import posixpath

import pandas as pd
from pandas import DataFrame

sep = posixpath.sep

from typing import Union, Iterable
import folium
import geopandas as gpd
from geopandas import GeoDataFrame
from pandas import IndexSlice as idx
from pluscodes.util import Decompositions

import numpy as np
from geopandas import GeoSeries

posixpath.sep = sep  # somehow, the geographic dependencies are deleting posixpath.sep
from pluscodes import tmatch

__all__ = ['Tessellation']


class Descriptor(abc.ABC):
    def __get__(self, instance: 'Tessellation', owner):
        self.tessellation = instance
        return self

    @abc.abstractmethod
    def __getitem__(self, item) -> 'Tessellation':
        ...


class DescriptorSpace(Descriptor):
    def __getitem__(self, item) -> 'Tessellation':
        t = self.tessellation
        tiles = t.tiles.loc[idx[:, item, :]]
        spaces: GeoDataFrame = t.spaces.loc[idx[:, item], :]
        iloc = spaces.index.get_level_values('iloc')
        gdf = t.gdf.iloc[iloc]
        return Tessellation(gdf, tiles, spaces)


class DescriptorIloc(Descriptor):
    def __getitem__(self, item) -> 'Tessellation':
        t = self.tessellation
        gdf = t.gdf.iloc[item]
        spaces = t.spaces.iloc[item, :]
        tiles = t.tiles.iloc[item, :, :]
        return Tessellation(gdf, tiles, spaces)


class DescriptorTile(Descriptor):
    def __getitem__(self, item) -> 'Tessellation':
        t = self.tessellation
        tiles: GeoSeries = self.tessellation.tiles.loc[idx[:, :, item]]
        iloc = tiles.index.get_level_values('iloc')
        spaces = t.spaces.loc[idx[iloc, :], :]
        gdf = t.gdf.iloc[iloc]
        return Tessellation(gdf, tiles, spaces)


class DescriptorCx(Descriptor):
    def __getitem__(self, item) -> 'Tessellation':
        t = self.tessellation
        spaces = t.spaces.cx[item]
        iloc = spaces.index.get_level_values('iloc')
        tiles = t.tiles.loc[idx[iloc, :, :], :]
        gdf = t.gdf.iloc[iloc]
        return Tessellation(gdf, tiles, spaces, )


# TODO: Fill out the convenience descriptors, and also create a Compare container class
# TODO: Perhaps monkey patch the tiles and spaces objects so that slicing them returns a Tessellation

class Tessellation:
    """
    Tesselation is a wrapper to aid with visualization and analytics regarding tjoin
    """
    iloc = DescriptorIloc()
    space = DescriptorSpace()
    tile = DescriptorTile()
    cx = DescriptorCx()

    def __init__(self, gdf: GeoDataFrame, tiles: GeoSeries, spaces: GeoDataFrame, name: str = None):
        if len(name) > 8:
            raise ValueError('Name must be no more than 8 characters')
        self.gdf = gdf
        self.tiles = tiles
        self.spaces = spaces
        self.name = name

    @classmethod
    # def from_gdf( cls, gdf: Union[GeoDataFrame, GeoSeries, str], geo: bool = False ):
    def from_gdf(
            cls,
            gdf: Union[GeoDataFrame, GeoSeries],
            name: str,
            *args,
            geo: bool = False,
    ):
        """

        :param gdf:
        :param args:
        :return:
        """
        dc = Decompositions(gdf)
        tiles = dc.tiles(geo=geo)
        spaces = dc.spaces(geo=geo)
        return cls(gdf, tiles, spaces, name)

    @classmethod
    def from_file(
            cls,
            filepath: str,
            name: str,
            *args,
            geo: bool = False,
    ):
        extension = filepath.rpartition('.')[-1]
        if extension == 'feather':
            gdf = gpd.read_feather(filepath)
        elif extension == 'parquet':
            gdf = gpd.read_parquet(filepath)
        else:
            gdf = gpd.read_file(filepath)
        return cls.from_gdf(gdf, name, geo=geo)

    def __len__(self):
        return len(self.tiles)

    def __repr__(self):
        bounds = ', '.join(
            str(val)
            for val in self.gdf.total_bounds.round(2)
        )
        return f'{self.__class__.__qualname__} {self.name} [{bounds}]'

    def explore_spaces(self, *args, **kwargs) -> folium.Map:
        if not isinstance(self.tiles, GeoSeries):
            raise TypeError('Tiles must be a GeoSeries')
        if not isinstance(self.spaces, GeoDataFrame):
            raise TypeError('Spaces must be a GeoDataFrame')
        if not isinstance(self.gdf, GeoDataFrame):
            raise TypeError('GDF must be a GeoDataFrame')

        spaces = self.spaces
        gdf = self.gdf
        m = folium.Map(
            location=gdf.geometry.iloc[0].centroid.coords[0][::-1],
            zoom_start=16,
        )

        gdf.boundary.explore(
            m=m,
            color='black',
            style_kwds=dict(
                fill=False,
            ),
            # **kwargs
        )
        spaces.explore(
            m=m,
            color='green',
            **kwargs
        )
        return m

    def explore_tiles(self, *args, **kwargs) -> folium.Map:
        if not isinstance(self.tiles, GeoSeries):
            raise TypeError('Tiles must be a GeoSeries')
        if not isinstance(self.spaces, GeoDataFrame):
            raise TypeError('Spaces must be a GeoDataFrame')
        if not isinstance(self.gdf, GeoDataFrame):
            raise TypeError('GDF must be a GeoDataFrame')

        tiles = self.tiles
        gdf = self.gdf
        m = folium.Map(
            location=gdf.geometry.iloc[0].centroid.coords[0][::-1],
            zoom_start=16,
        )
        gdf.explore(
            m=m,
            color='black',
            style_kwds=dict(
                fill=False,
            ),
            # **kwargs
        )
        tiles.explore(
            m=m,
            color='green',
            **kwargs
        )
        return m

    def _tconcat(self, other: 'Tessellation') -> GeoDataFrame:
        # TODO: categorical string instead of unicode dtpe
        iloc_left = tmatch._match_spaces(self.tiles, other.tiles)
        spaces = other.spaces.droplevel('iloc')
        spaces = spaces.merge(iloc_left, left_on='space', right_index=True, how='right', suffixes=None)
        spaces['name'] = pd.Categorical([other.name] * len(spaces))
        spaces = spaces.set_index(['iloc', 'name'], append=True)
        return spaces

    def tconcat(self, other: Union['Tessellation', list['Tessellation']]) -> DataFrame:
        spaces = self.spaces.copy()
        if isinstance(other, Tessellation):
            other = [other]
        names = [self.name, *(o.name for o in other)]
        spaces['name'] = pd.Categorical([self.name] * len(spaces), categories=names)
        spaces = spaces.set_index('name', append=True)
        dataframes = itertools.chain((spaces,), (self._tconcat(o) for o in other))
        concat = pd.concat( dataframes, )
        concat = concat.reorder_levels(['iloc', 'name', 'space'])
        return concat


if __name__ == '__main__':
    test = Tessellation.from_file('/home/arstneio/Downloads/ne.feather', 'test', geo=False)
    test2 = Tessellation.from_file('/home/arstneio/Downloads/ne.feather', 'test2', geo=False)
    result = test.tconcat(test2)
    # test._tconcat(test2)
