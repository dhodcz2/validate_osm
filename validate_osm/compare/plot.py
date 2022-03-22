import itertools
import warnings

import matplotlib.colors as mcolors
from typing import Hashable, Optional, Iterable, Union

import pandas as pd

COLORS = list(mcolors.TABLEAU_COLORS.values())
HATCHES = '\ - | \\'.split()

import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np


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
        ...

    @figsize.setter
    def figsize(self, val: tuple[float, float]):
        plt.rcParams['figure.figsize'] = val

    @dark_background.setter
    def dark_background(self, val: bool):
        if val:
            plt.style.use('dark_background')
        else:
            plt.style.use('default')

    def matches(self, ubid=None, annotations: bool = False):
        from validate_osm.compare.compare import Compare
        self._instance: Compare
        agg = self._instance.aggregate.copy()
        if ubid is None:
            pass
        elif isinstance(ubid, Hashable):
            agg = agg.xs(ubid, level='ubid')
        elif isinstance(ubid, Iterable):
            agg = agg[agg.index.isin(set(ubid))]
        else:
            raise TypeError(ubid)

        names = list(agg.index.get_level_values('name').unique())
        fig, axes = plt.subplots(1, len(names))

        # Assign a color and hatch to every unique UBID
        lenc = len(COLORS)
        lenh = len(HATCHES)
        groups = agg.groupby('ubid').indices.values()
        # agg['color'] = [
        #     COLORS[i % lenc]
        #     for i, group in enumerate(agg.groupby('ubid').groups.values())
        #     for val in group
        # ]
        # agg['hatch'] = [
        #     HATCHES[i % lenh]
        #     for i, group in enumerate(groups)
        #     for val in group
        # ]
        agg['color'] = pd.Series((
            COLORS[i % lenc]
            for i, group in enumerate(groups)
            for val in group
        ), index=iter(groups))
        agg['hatch'] = pd.Series((
            HATCHES[i % lenc]
            for i, group in enumerate(groups)
            for val in group
        ))

        # Each name corresponds to an axis; for each unique UBID with that name, plot with the color and hatch
        for name, axis in zip(names, axes):
            axis.set_title(name)
            subagg: gpd.GeoDataFrame = agg.xs(name, level='name')
            for (color, hatch), loc in subagg.groupby(['color', 'hatch']).groups.items():
                # subagg.loc[loc].geometry.plot(color=color, hatch=hatch, ax=axis)
                try:
                    subagg.loc[loc].geometry.plot(color=color, hatch=hatch, ax=axis)
                except TypeError as e:
                    print(f"{loc=}, {color=}, {hatch=}, {axis=}, {name=}")
                    raise e
            if annotations:
                for centroid, iloc in zip(subagg['centroid'], subagg['iloc']):
                    axis.annotate(str(iloc), xy=(float(centroid.x), float(centroid.y)))

    def matched(self, name: Hashable, others: Optional[Hashable] = None):
        from validate_osm.compare.compare import Compare
        self._instance: Compare
        gdf = self._instance.percent_overlap(name, others)
        ax = gdf.plot(cmap='RdYlGn', column='intersection')
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
        if not isinstance(agg, gpd.GeoDataFrame):
            raise TypeError(agg)
        if not isinstance(data, gpd.GeoDataFrame):
            raise TypeError(data)

        fig, (axd, axa) = plt.subplots(1, 2)
        axd.set_title(f'{name}.data')
        axa.set_title(f'{name}.aggregate')

        # Assign a color and hatch to every UBID
        lenc = len(COLORS)
        lenh = len(HATCHES)
        ubids = agg.index.get_level_values('ubid')
        ubids = list(ubids)
        for df in (data, agg):
            indices = [
                df.xs(ubid, level='ubid', drop_level=False).index
                for ubid in ubids
            ]
            df['color'] = pd.Series((
                COLORS[i % lenc]
                for i, index in enumerate(indices)
                for _ in range(len(index))
            ), index=itertools.chain.from_iterable(indices))
            df['hatch'] = pd.Series((
                HATCHES[i % lenh]
                for i, index in enumerate(indices)
                for _ in range(len(index))
            ), index=itertools.chain.from_iterable(indices))

        # Each name corresponds to an axis; for each unique UBID with that name; plot with the color and hatch
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

    def where(self, validating: Hashable, name: Hashable):
        """
        Plots where the discrepancies; shows a colormap of percent_error and highlights entries that
        the average value.
        :param validating:
        :param name:
        :return:
        """
