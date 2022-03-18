import geopandas as gpd
from geopandas import GeoDataFrame
import pandas as pd
from typing import Type, Iterable, Iterator

from annoy import AnnoyIndex
from typing import Collection, Optional

import matplotlib.pyplot as plt
import pandas

import pandas as pd
import geopandas as gpd
from pandas import DataFrame, Series
from geopandas import GeoDataFrame, GeoSeries
import matplotlib.pyplot as plt

plt.style.use('dark_background')
idx = pandas.IndexSlice

import functools

from pandas import IndexSlice as idx
import pandas as pd
from typing import Type
from pandas import DataFrame, Series
import geopandas as gpd
from validate_osm.source.source import Source


class Where:
    """
    If n is a Needle, returns index of the abstraction.
    If n is not a Needle, returns index of the source.
    """

    def __get__(self, instance, owner):
        self.compare: 'Compare' = instance
        self.compare.plot: 'Plot'
        self.compare.where: 'Where'
        self.compare.values: 'Values'
        ...

    def has(self, ways, relations):
        ...

    def within(self, bbox):
        ...

    def worst(self, count):
        ...

    def best(self, count):
        ...

    def outlier(self, k):
        ...

    def region(self, i, n):
        ...

    def near(self, i, n, k):
        ...

    def ungrouped(self, n):
        ...


class Values:
    def __get__(self, instance: 'Compare', owner):
        self.compare: 'Compare' = instance
        self.compare.plot: 'Plot'
        self.compare.where: 'Where'
        self.compare.values: "Values"

        # indices = (
        #     self.compare.xs(n, level='n').index
        #     for n in self.data.keys()
        # )
        # # TODO: Hopefully this is overwritten with each each call to getattr(values, _)
        # self.memo: dict[str, dict[str, DataFrame]] = {
        #     n: {
        #         n: pd.DataFrame(
        #             pd.Series(0, name=v, index=i)
        #             for v in self.validating
        #         )
        #     }
        #     for n, i in zip(self.data.keys(), indices)
        # }
        #

    def oned(self, left: str, right: str, how='percent') -> DataFrame:
        gdf_left: GeoDataFrame = self.compare.gdf.loc[idx[:, :, left], :]
        gdf_right: GeoDataFrame = self.compare.gdf.loc[idx[:, :, right], :]
        i = (
            gdf_left.index.get_level_values('i')
                .intersection(gdf_right.index.get_level_values('i'))
        )
        if how == 'percent':
            result = {
                v: (
                    for l, r
                )
                for v in self.compare.validating
            }
        elif how == 'abs':
            ...
        else:
            raise ValueError(how)
        Compares
        one
        Source
        with another

    # gdf_left: GeoDataFrame = self.compare.gdf.loc[idx[:, ]]

    # if left in self.memo and right in self.memo[left]:
    #     return self.memo[left][right]
    # loc = pd.Index.union(
    #     self.compare.xs(left, level='name').index,
    #     self.compare.xs(right, level='name').index
    # )
    # gdf_left: DataFrame = self.compare.loc[idx[loc, left]]
    # gdf_right: DataFrame = self.compare.loc[idx[loc, right]]
    # if how == 'percent':
    #     result = DataFrame({
    #         v: {
    #             (r - l) / r
    #             if r > l else
    #             (l - r) / l
    #             for l, r in zip(gdf_left[v], gdf_right[v])
    #         }
    #         for v in self.validating
    #     }, index=loc)
    # elif how == 'abs':
    #     result = DataFrame({
    #         v: {
    #             abs(l - r)
    #             for l, r in zip(gdf_left[v], gdf_right[v])
    #         }
    #         for v in self.validating
    #     }, index=loc)
    # else:
    #     raise ValueError(how)
    # self.memo[left][right] = result
    # self.memo[right][left] = result
    # return result
    #

    def twod(self, defendant: str, how='percent') -> DataFrame:
        # MultiIndex(index, validating) -> [name]
        # df.loc[idx[i, v], n] = #
        """
        index: {
            height: { lod, 3dm }
            floors: { lod, 3dm }
        }
        """
        others = set(self.sources.keys())
        others.discard(defendant)
        result = pd.concat((
            self.oned(defendant, other, how)
                .unstack()
                .swaplevel(0, 1)
                .rename(other)
            for other in others
        ), axis=1)
        result = result.sort_index()
        return result

    @functools.lru_cache(2)
    def threed(self, how='percent') -> DataFrame:
        # MultiIndex(index, validating, name) -> [name]
        # df.loc[idx[i, v, n]] = #
        """
        index:  {
            height: {
                lod: {lod, 3dm, osm}
                3dm: {lod, 3dm, osm}
                osm: {lod, 3dm, osm}
            },
            floors: {
                lod: {lod, 3dm, osm}
                3dm: {lod, 3dm, osm}
                osm: {lod, 3dm, osm}
            }
        }
        """
        result = pd.concat((
            self.twod(defendant, how)
            for defendant in self.sources.keys()
        ), axis=1)
        # TOOD: Instead of a memoization within oned or twod which aren't personally useful for those functions,
        #   wrap the functions locally.


class Plot:
    # Problem: Because of import
    def __get__(self, instance: object, owner: type):
        self.compare: Compare = instance
        self.compare.values: 'Values'
        self.compare.plot: 'Plot'
        self.compare.where: 'Where'

    def what(
            self,
            v: str,
            n: str,
            i: Optional[Collection[int]] = None
    ):
        """
        Plots original values for a dataset.
        :param n:   The name of the particular Source to be investigated
        :param v:   The Validating data to be investigated
        :param i:   The Abstraction indices to be investigated; all if None
        :return:
        """
        gdf = self.gdf.loc[idx[i, v, n], n]

    # def where(self, v: str, i: Optional[Collection[int]] = None, n: Optional[Collection[str]] = None):
    def where(
            self,
            v: str,
            i: Collection[int] = None,
            n: Collection[str] = None
    ):
        """
        Plots where the discrepancies are;
        shows a colormap of the percent_error and highlights entries that deviate
        from the local average.
        :param v:   The Validating column for which the compare will be made
        :param i:   The Abstraction indices for which the compare will be made, all if None
        :param n:   The Names of which particular Sources the osm aggregate will be compared with, all if None
        :return:
        """
        # TODO: What if the Source doesn't have geometry?
        # if n is not None:
        #     n = set(n).add('osm')
        if n is not None:

        gdf: GeoDataFrame = self.gdf.loc[idx[i, v, n], n]

        fig, ax = plt.subplots()
        ax.tick_params(labelsize=100)

        if len(gdf):
            gdf.plot(ax=ax, cmap='rainbow', column=col, legend=True, )
            gdf.exterior.plot(ax=ax, color='black')

        annoy = AnnoyIndex(2, 'euclidean')
        # TODO: If multiple n, determine the average
        mean: Series = gdf.aggregate()
        for i, centroid in enumerate(gdf['centroid']):
            annoy.add_item(i, (centroid.x, centroid.y))

    def how(
            self,
            n: Collection[str],
            v: Optional[Collection[str]],
            i: Optional[Collection[int]] = None,

    ):
        """
        Plots how the value was formed for entries;
        shows the Source values that comprise an Abstraction entry
        :param n:   The name of the particular Source to be investigated
        :param i:   The Abstraction indices to be investigated; all if None
        :param v:   The Validating data to be investigated
        :return:
        """
        gdf = self.gdf.loc[idx[i, v, n], n]

    def why(self, i: Collection[int], v: Collection[str], n: Collection[str], ):
        """
        Plots why the aggregate were made;
        shows the Needles priming the Source entries into an Abstraction entry
        :param i:   The Abstraction indices to be investigated; all if None
        :param v:   The Validating data to be investigated;
        :param n:   The name of the particular Haystack to be investigated
        :return:
        """
        gdf = self.gdf.loc[idx[i, v, n], n]


class Compare:
    values: Values = Values()
    plot: Plot = Plot()
    where: Where = Where()

    def __init__(self, sources: Iterable[Type[Source]]):
        """
        :param sources:
        """
        """
        self.gdf.loc[idx[i, v, n],:]
        index: {
            height: {osm, 3dm, lod}
            floors: {osm, 3dm, lod}
        }
        """
        self.values: Values
        self.plot: Plot
        self.where: Where
        self.sources = set(sources)
        self.validating: set[str] = {
            validating
            for source in sources
            for validating in source.validating_types.keys()
        }

        gdfs: Iterator[GeoDataFrame] = (
            source
                .from_abstraction()
                .__getitem__(self.validating)
                .unstack()
                .swaplevel(0, 1)
                .assign(n=source.name)
                .set_index('n', append=True)
            for source in sources
        )
        # TODO: I have taken concat out of Source because it only seems to be needed in Source.data;
        #   Compare should return DataFrames and not GeoDataFrames
        # self.gdf: GeoDataFrame = Source.concat(gdfs)


Compare.values: Values
Compare.plot: Plot
Compare.where: Where
