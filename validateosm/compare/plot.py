import warnings
import matplotlib.axis
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import geopandas as gpd

plt.style.use('dark_background')


class DescriptorPlot:
    def __get__(self, instance, owner):
        self._instance = instance
        self._owner = owner
        warnings.warn("plt.rcParams['figure.figsize'] = (..., ...)")
        return self

    def matches(self, ubid=None):
        from validateosm.compare.compare import Compare
        self._instance: Compare
        agg = self._instance.aggregate
        names = list(agg.index.get_level_values('name').unique())
        fig, axes = plt.subplots(1, len(names))

        if ubid is None:
            import matplotlib.colors as mcolors
            colors = list(mcolors.TABLEAU_COLORS.values())
            hatches = '\ - | \\'.split()

            # Assign a color and hatch to every unique UBID
            agg = agg.assign(color=np.nan, hatch=np.nan)
            for i, index in enumerate(agg.groupby('ubid').groups.values()):
                agg.loc[index, 'color'] = colors[i % len(colors)]
                agg.loc[index, 'hatch'] = hatches[i % len(hatches)]

            # Each name corresponds to an axis; for each unique UBID with that name, plot with the color and hatch
            for name, axis in zip(names, axes):
                axis.set_title(name)
                subagg: gpd.GeoDataFrame = agg.xs(name, level='name')
                for (color, hatch), loc in subagg.groupby(['color', 'hatch']).groups.items():
                    subagg.loc[loc].geometry.plot(color=color, hatch=hatch, ax=axis)
                for centroid, iloc in zip(subagg['centroid'].to_crs(4326), subagg['iloc']):
                    axis.annotate(str(iloc), xy=(float(centroid.x), float(centroid.y)))




        else:
            raise NotImplementedError
            # Show the match overlapping.
            ...

    def matched(self, name, ubid=None):
        from validateosm.compare.compare import Compare
        self._instance: Compare
        agg: gpd.GeoDataFrame = self._instance.aggregate
        this: gpd.GeoDataFrame = self._instance.aggregate.xs(name, level='name')

        if ubid is None:
            ubids = this.groupby('ubid').indices.values()



        else:
            raise NotImplementedError
