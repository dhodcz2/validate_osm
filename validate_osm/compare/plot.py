import itertools

import matplotlib.colors as mcolors
from typing import Hashable, Optional, Iterable, Union

import pandas as pd

COLORS = list(mcolors.TABLEAU_COLORS.values())
HATCHES = '\ - | \\'.split()

import geopandas as gpd
import matplotlib.pyplot as plt


def _pseudo_colormap(groups: Iterable[pd.Index], gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    # Assign a color and hatch to every unique UBID
    lenc = len(COLORS)
    lenh = len(HATCHES)
    # index = pd.MultiIndex(itertools.chain.from_iterable(groups), names=gdf.index.names)
    index = pd.MultiIndex.from_tuples(itertools.chain.from_iterable(groups), names=gdf.index.names)
    color = pd.Series((
        COLORS[i % lenc]
        for i, group in enumerate(groups)
        for _ in group
    ), index=index)
    hatch = pd.Series((
        COLORS[i % lenh]
        for i, group in enumerate(groups)
        for _ in group
    ), index=index)
    gdf = gdf.assign(color=color, hatch=hatch)
    return gdf


class DescriptorPlot:
    def __get__(self, instance, owner):
        self._instance = instance
        self._owner = owner
        return self

    @property
    def dark_background(self):
        return None

    @property
    def figsize(self):
        return plt.rcParams['figure.figsize']

    @figsize.setter
    def figsize(self, val: tuple[float, float]):
        plt.rcParams['figure.figsize'] = val

    @figsize.deleter
    def figsize(self):
        del plt.rcParams['figure.figsize']

    @dark_background.setter
    def dark_background(self, val: bool):
        if val:
            plt.style.use('dark_background')
        else:
            plt.style.use('default')

    @staticmethod
    def matches(self, ubid=None, annotation: Optional[str] = 'iloc'):
        from validate_osm.compare.compare import Compare
        self._instance: Compare
        agg = self._instance.aggregate

        if ubid is None:
            pass
        elif isinstance(ubid, Hashable):
            agg = agg.xs(ubid, level='ubid')
        elif isinstance(ubid, Iterable):
            agg = agg[agg.index.isin(set(ubid))]
        else:
            raise TypeError(ubid)

        agg: gpd.GeoDataFrame
        groups: Iterable[pd.Index] = agg.groupby('ubid').groups.values()
        agg = _pseudo_colormap(groups, agg)

        names = list(agg.index.get_level_values('name').unique())
        fig, axes = plt.subplots(1, len(names))

        # Each name corresponds to an axis; for each unique UBID with that name, plot with the color and hatch
        for name, axis in zip(names, axes):
            axis.set_title(f'{name}.aggregate')
            subagg: gpd.GeoDataFrame = agg.xs(name, level='name')
            for (color, hatch), loc in subagg.groupby(['color', 'hatch']).groups.items():
                subagg.loc[loc].geometry.plot(color=color, hatch=hatch, ax=axis)
            if annotation:
                for centroid, iloc in zip(subagg['centroid'], subagg['iloc']):
                    axis.annotate(str(iloc), xy=(float(centroid.x), float(centroid.y)))

    def matched(self, name: Hashable, others: Optional[Hashable] = None, annotation: Optional[str] = 'iloc'):
        from validate_osm.compare.compare import Compare
        self._instance: Compare
        gdf = self._instance.percent_overlap(name, others)
        ax = gdf.plot(cmap='RdYlGn', column='intersection')
        if annotation:
            for centroid, iloc in zip(gdf['centroid'], gdf['iloc']):
                ax.annotate(str(iloc), xy=(float(centroid.x), float(centroid.y)))

    def how(self, name: str, column: Optional[Hashable] = None, ubid: Union[None, Hashable, Iterable[Hashable]] = None):
        """
        Plots how data was grouped to form an aggregate
        :param name:    The Source that is being investigated
        :param column:  The value that is being inspected; shows iloc if None
        :param ubid:
        :return:
        """
        from validate_osm.compare.compare import Compare
        self._instance: Compare
        agg = self._instance.aggregate.xs(name, level='name', drop_level=False)
        data = self._instance.data.xs(name, level='name', drop_level=False)
        if ubid is None:
            pass
        elif isinstance(ubid, Hashable):
            agg = agg.xs(ubid, level='ubid', drop_level=False)
            data = data.xs(ubid, level='ubid', drop_level=False)
        elif isinstance(ubid, Iterable):
            ubid = set(ubid)
            agg = agg[agg.index.isin(ubid)]
            data = data[data.index.isin(ubid)]
        else:
            raise TypeError(ubid)
        agg: gpd.GeoDataFrame
        data: gpd.GeoDataFrame

        ubids = agg.groupby('ubid').groups.values()
        agg = _pseudo_colormap(ubids, agg)
        ubids = data.groupby('ubid').groups.values()
        data = _pseudo_colormap(ubids, data)

        fig, (axd, axa) = plt.subplots(1, 2)
        axd.set_title(f'{name}.data')
        axa.set_title(f'{name}.aggregate')
        if column is None:
            column = 'iloc'
        for df, axis in zip((data, agg), (axd, axa)):
            axis.set_xticklabels([])
            axis.set_yticklabels([])
            axis.set_xticks([])
            axis.set_yticks([])
            for (color, hatch), loc in df.groupby(['color', 'hatch']).groups.items():
                df.loc[loc].geometry.boundary.plot(color=color, hatch=hatch, ax=axis)
            for centroid, value in zip(df['centroid'], df[column]):
                axis.annotate(str(value), xy=(float(centroid.x), float(centroid.y)))

    def where(self, name: str, value: str, annotation: Optional[str] = 'iloc'):
        """
        Plots where the discrepancies; shows a colormap of percent_error and highlights entries that
        the average value.
        :param validating:
        :param name:
        :return:
        """


