import math

from geopandas import GeoDataFrame
import itertools

import matplotlib.colors as mcolors
from typing import Hashable, Optional, Iterable, Union, Iterator, Any, Callable

import pandas as pd

COLORS = list(mcolors.TABLEAU_COLORS.values())
HATCHES = '\\ - | /'.split()

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
        HATCHES[i % lenh]
        for i, group in enumerate(groups)
        for _ in group
    ), index=index)
    gdf = gdf.assign(color=color, hatch=hatch)
    return gdf


def _suptitle(compare, fig, funcname, l: dict):
    name = f'{compare.name}.plot.{funcname}'
    l.pop('self')
    msg = ', '.join(f'{key}={val}' for key, val in l.items())
    fig.suptitle(f'{name}({msg})')


def _annotate(ax, params, gdf):
    if (annotation := params['annotation']):
        for centroid, v in zip(gdf['centroid'].to_crs(params['crs']), gdf[annotation]):
            if isinstance(v, float):
                v = '%.1f' % v
            else:
                v = str(v)
            ax.annotate(v, xy=(float(centroid.x), float(centroid.y)))


class DescriptorPlot:
    def __init__(self):
        self.style = 'dark_background'

    def __get__(self, instance, owner):
        self._instance = instance
        self._owner = owner
        return self

    @property
    def xs(self) -> Callable[[Hashable], GeoDataFrame]:
        return self._instance.xs

    @property
    def figsize(self):
        return plt.rcParams['figure.figsize']

    @figsize.setter
    def figsize(self, val: tuple[float, float]):
        plt.rcParams['figure.figsize'] = val

    @figsize.deleter
    def figsize(self):
        del plt.rcParams['figure.figsize']

    @property
    def style(self):
        raise NotImplementedError

    @style.setter
    def style(self, val):
        plt.style.use(val)

    @style.deleter
    def style(self):
        plt.style.use('default')

    @property
    def params(self) -> dict[str, Any]:
        if not hasattr(self, '_params'):
            self._params = {
                'crs': 3857,
                'annotation': 'iloc',

            }
        return self._params

    def matches(self, **kwargs):
        local = locals().copy()
        (params := self.params.copy()).update(kwargs)
        from validate_osm.compare.compare import Compare
        self._instance: Compare
        agg = self._instance.aggregate

        # if ubid is None:
        #     pass
        # elif isinstance(ubid, Hashable):
        #     agg = agg.xs(ubid, level='ubid')
        # elif isinstance(ubid, Iterable):
        #     agg = agg[agg.index.isin(set(ubid))]
        # else:
        #     raise TypeError(ubid)

        names = list(agg.index.get_level_values('name').unique())
        fig, axes = plt.subplots(1, len(names))
        print(local)
        _suptitle(self._instance, fig, self.matches.__name__, local)

        agg: gpd.GeoDataFrame
        groups: Iterable[pd.Index] = agg.groupby('ubid').groups.values()
        agg = _pseudo_colormap(groups, agg)

        # Each name corresponds to an axis; for each unique UBID with that name, plot with the color and hatch
        for name, axis in zip(names, axes):
            axis.set_title(f'{name}.aggregate')
            subagg: gpd.GeoDataFrame = agg.xs(name, level='name')
            for (color, hatch), loc in subagg.groupby(['color', 'hatch']).groups.items():
                subagg.loc[loc, 'geometry'] \
                    .to_crs(params['crs']) \
                    .plot(color=color, hatch=hatch, ax=axis)
            _annotate(axis, params, subagg)
            # if (annotation := params['annotation']):
            #     for centroid, iloc in zip(subagg['centroid'].to_crs(params['crs']), subagg[annotation]):
            #         axis.annotate(str(iloc), xy=(float(centroid.x), float(centroid.y)))

    def containment(self, of, within, **kwargs):
        locale = locals()
        (params := self.params.copy()).update(kwargs)
        fig, ax = plt.subplots(1, 1)
        _suptitle(self._instance, fig, self.containment.__name__, locale)
        ax.set_title(f'{of}.aggregate (unmatched grey); {within}.aggregate boundary;')

        from validate_osm.compare.compare import Compare
        compare: Compare = self._instance

        of = self.xs(of)
        within = self.xs(within)
        within = within[within.index.isin(set(of.index))]
        overlap = compare.containment(of, within)
        of = of.assign(overlap=overlap)

        of[of['overlap'].notna()].plot(cmap='RdYlGn', column='overlap', ax=ax, legend=True)
        of[of['overlap'].isna()].plot(color='gray', ax=ax)
        within.geometry.boundary.plot(ax=ax)
        #

        if (annotation := params['annotation']):
            of = of.sort_values('overlap', ascending=True)
            bottom_5 = math.floor(len(of) * .05)
            of = of.iloc[:bottom_5]
            for centroid, iloc in zip(of['centroid'], of[annotation]):
                ax.annotate(str(iloc), xy=(float(centroid.x), float(centroid.y)), color='blue', fontsize=14)

    # TODO: Perpetrator is largest completion in a footprint

    def completion(self, of, **kwargs):
        locale = locals()
        (params := self.params.copy()).update(kwargs)
        fig, ax = plt.subplots(1, 1)
        _suptitle(self._instance, fig, self.completion.__name__, locale)
        ax.set_title(f'completion of {of}.aggregate within footprints')

        from validate_osm.compare.compare import Compare
        compare: Compare = self._instance

        of = self.xs(of)
        of = of.assign(completion=compare.completion(of))

        # of.plot(cmap='RdYlGn', column='completion', ax=ax, legend=True)

        of.to_crs(params['crs']).plot(cmap='RdYlGn', column='completion', ax=ax, legend=True)
        # of.geometry.to_crs(params['crs']).plot(cmap='RdYlGn', column='completion', ax=ax, legend=True)

        footprints = compare.footprints
        footprints = footprints[footprints.index.isin(set(of.index.get_level_values('ubid')))]
        footprints.geometry.to_crs(params['crs']).boundary.plot(ax=ax)

        _annotate(ax, params, of)

    def quality(self, of, and_, **kwargs):
        locale = locals()
        from validate_osm.compare.compare import Compare
        compare: Compare = self._instance

        (params := self.params.copy()).update(kwargs)
        fig, ax = plt.subplots(1, 1)
        _suptitle(self._instance, fig, self._instance.plot.quality.__name__, locale)
        ax.set_title(f'intersection area of {of} and {and_}')

        from validate_osm.compare.compare import Compare
        compare: Compare = self._instance

        union = compare.union(of, and_)
        intersection = compare.intersection(of, and_)
        quality = compare.containment(of=union, in_=intersection)

        gdf = GeoDataFrame({
            'geometry': intersection,
            'quality': quality,
            'centroid': intersection.centroid,
            # 'iloc': compare.footprints['iloc']
        })
        gdf['iloc'] = compare.footprints['iloc']

        gdf.plot(cmap='RdYlGn', column='quality', ax=ax, legend=True)
        union.boundary.plot(ax=ax)
        _annotate(ax, params, gdf)

    def difference_percent(self, of, and_, values, **kwargs):
        locale = locals()
        (params := self.params.copy()).update(kwargs)
        from validate_osm.compare import Compare
        compare: Compare = self._instance
        fig, ax = plt.subplots(1, 1)
        _suptitle(compare, fig, compare.plot.difference_percent.__name__, locale)

        if not isinstance(of, str) or not isinstance(and_, str):
            raise TypeError

        difference = compare.matrix.percent_difference(rows=of, columns=and_, values=values)
        difference = difference.reset_index('name', drop=True)[and_]
        # difference = difference.reset_index(compar, level='name', drop=True)[and_]
        union = compare.union(of, and_).loc[difference.index]
        intersection = compare.intersection(of, and_).loc[difference.index]
        iloc = compare.footprints.loc[difference.index, 'iloc']

        gdf = GeoDataFrame({
            'geometry': intersection,
            'difference': difference,
            'centroid': intersection.centroid,
            'iloc': iloc
        })

        gdf.plot(cmap='RdYlGn_r', column='difference', ax=ax, legend=True)
        union.boundary.plot(ax=ax)
        _annotate(ax, params, gdf)

        # difference = compare.difference_percent(of, and_, value)
        # union = compare.union(of, and_).loc[difference.index]
        # intersection = compare.intersection(of, and_).loc[difference.index]
        #
        # gdf = GeoDataFrame({
        #     'geometry': intersection,
        #     'difference': difference,
        #     'centroid': intersection.centroid
        # })
        # gdf['iloc'] = compare.footprints['iloc']
        #
        # gdf.plot(cmap='RdYlGn_r', column='difference', ax=ax, legend=True)
        # union.boundary.plot(ax=ax)
        # _annotate(ax, params, gdf)
        #

    def difference_scaled(self, of, and_, value, **kwargs):
        locale = locals()
        (params := self.params.copy()).update(kwargs)
        from validate_osm.compare import Compare
        compare: Compare = self._instance
        fig, ax = plt.subplots(1, 1)
        _suptitle(compare, fig, compare.plot.difference_scaled.__name__, locale)

        difference = compare.difference_scaled(of, and_, value)
        union = compare.union(of, and_).loc[difference.index]
        intersection = compare.intersection(of, and_).loc[difference.index]

        gdf = GeoDataFrame({
            'geometry': intersection,
            'difference': difference,
            'centroid': intersection.centroid
        })
        gdf['iloc'] = compare.footprints['iloc']

        gdf.plot(cmap='RdYlGn_r', column='difference', ax=ax, legend=True)
        union.boundary.plot(ax=ax),
        _annotate(ax, params, gdf)

        # difference = compare.difference_scaled(of, and_, value)
        # footprints: GeoDataFrame = compare.footprints.loc[difference.index].assign(
        #     difference=difference
        # )
        #
        # footprints.to_crs(params['crs']).plot(cmap='RdYlGn_r', column='difference', ax=ax, legend=True)
        # _annotate(ax, params, footprints)
        #

    def matched(self, name: Hashable, others: Optional[Hashable] = None, annotation: Optional[str] = 'iloc'):
        local = locals()
        fig, ax = plt.subplots(1, 1)
        _suptitle(self._instance, fig, self.matched.__name__, local)
        ax.set_title(f'{name}.agg ')
        from validate_osm.compare.compare import Compare
        self._instance: Compare
        gdf = self._instance.percent_overlap_of_aggregate(name, others)
        gdf.plot(cmap='RdYlGn', column='intersection', ax=ax)

        if annotation:
            for centroid, iloc in zip(gdf['centroid'], gdf[annotation]):
                ax.annotate(str(iloc), xy=(float(centroid.x), float(centroid.y)))

    def how(self, name: str, column: Optional[Hashable] = None, ubid: Union[None, Hashable, Iterable[Hashable]] = None):
        """
        Plots how data was grouped to form an aggregate
        :param name:    The Source that is being investigated
        :param column:  The value that is being inspected; shows iloc if None
        :param ubid:
        :return:
        """
        fig, (axd, axa) = plt.subplots(1, 2)
        _suptitle(self._instance, fig, self.how.__name__, locals())
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

    def where(self, column: str, names: Union[str, Iterable[str]] = None, annotation: Optional[str] = 'iloc'):
        # TODO: if no comparison, color is gray
        local = locals()

        from validate_osm.compare.compare import Compare
        self._instance: Compare
        if isinstance(names, str):
            names = (names,)
        elif names is None:
            names = list(self._instance.sources.keys())
        else:
            names = list(names)

        fig, axes = plt.subplots(1, len(names))
        _suptitle(self._instance, fig, self.where.__name__, local)

        agg = self._instance.aggregate
        subaggs: dict[str, gpd.GeoDataFrame] = {
            name: agg.xs(name, level='name', drop_level=False)
            for name in names
        }

        for (name, subagg), ax in zip(subaggs.items(), axes):
            subagg: GeoDataFrame
            others: gpd.GeoDataFrame = agg[agg.identity.get_level_values('name') != name]
            others = others[others[column].notna()]

            ubid = set(
                subagg[subagg[column].notna()].identity.get_level_values('ubid')
                    .quality(others.index.get_level_values('ubid'))
            )

            colored: GeoDataFrame = subagg[subagg.index.get_level_values('ubid').isin(ubid)]
            grey: GeoDataFrame = subagg[~subagg.index.get_level_values('ubid').isin(ubid)]
            others: GeoDataFrame = others[others.index.get_level_values('ubid').isin(ubid)]

            other_values = {
                ubid: value
                for ubid, value in zip(others.index.get_level_values('ubid'), others[column])
            }

            def gen() -> Iterator[float]:
                nonlocal subagg
                nonlocal ubid
                nonlocal others
                for ubid, value in zip(colored.index.get_level_values('ubid'), colored[column]):
                    other = other_values[ubid]
                    if other > value:
                        yield (other - value) / other
                    else:
                        yield (value - other) / value

            colored['percent_error'] = [(1 - val) for val in gen()]

            ax.set_title(f'{name}.aggregate.{column}')
            colored.plot(ax=ax, column='percent_error', cmap='RdYlGn')
            grey.plot(color='grey', ax=ax)

            if annotation:
                for c, p, a in zip(colored['centroid'], colored['percent_error'], colored[annotation]):
                    if p < .50:
                        ax.annotate(str(a), xy=(float(c.x), float(c.y)), color='black')

    # def _aggregate_from_string(self, string: str) -> GeoDataFrame:
    #     if string == 'footprint' or string == 'footprints':
    #         return self._instance.footprints
    #     else:
    #         return self._instance.aggregate.xs(string, level='name', drop_level=True)
