from typing import Hashable
import numpy as np
from shapely.geometry.base import BaseGeometry
import geopandas as gpd
import pandas as pd
import shapely
from annoy import AnnoyIndex
import functools
from typing import ValuesView, Collection, Generator, Union, Type


class DescriptorFootprint:
    def __get__(self, instance, owner):
        from validateosm.compare.compare import Compare
        self._instance: Compare = instance
        self._owner: Type[Compare] = owner
        return self

    def __delete__(self, instance):
        del self.footprints

    def __len__(self):
        return len(self.footprints)

    @functools.cached_property
    def footprints(self) -> gpd.GeoSeries:
        # TODO: Why isn't len(footprints) < len(data)?
        data = self._instance.data[['geometry', 'centroid']]
        data['geometry'] = data['geometry'].to_crs(3857)
        data['centroid'] = data['centroid'].to_crs(3857)
        data['area'] = data.area
        data = data.sort_values(by='area', ascending=False)
        footprints = pd.Series(range(len(data)), index=data.index)

        annoy = AnnoyIndex(2, 'euclidean')
        for i, centroid in enumerate(data['centroid']):
            annoy.add_item(i, (centroid.x, centroid.y))  # lower i, higher area
        annoy.build(10)


        for i, (g, a) in enumerate(data[['geometry', 'area']].values):
            for n in annoy.get_nns_by_item(i, 30):
                if i <= n:  # neighbor is smaller or neighbor is the same item
                    continue
                footprint: pd.Series = data.iloc[n]
                geometry: shapely.geometry.base.BaseGeometry = footprint['geometry']
                if not geometry.intersects(g):
                    continue
                # TODO: It seems that there are some cases where intersection is below 50% but
                #   it is still appropriate.
                #   Perhaps this is still necessary to prevent against odd cases.
                if geometry.intersection(g).area / a < .1:
                    continue
                footprints.iloc[i] = footprints.iloc[n]
                break

        groupby = footprints.groupby(footprints)

        def geometry() -> Generator[shapely.geometry.Polygon, None, None]:
            geom = data['geometry']
            for iloc in groupby.indices.values():
                if len(iloc) == 1:
                    yield geom.iat[iloc[0]]
                else:
                    yield geom.iloc[iloc].unary_union

        def centroid() -> Generator[shapely.geometry.Point, None, None]:
            c = data['centroid']
            for iloc in groupby.indices.values():
                if len(iloc) == 1:
                    yield c.iat[iloc[0]]
                else:
                    yield c.iloc[iloc].unary_union.centroid

        # footprints = gpd.GeoDataFrame({'geometry': geometry(), 'centroid': centroid()})
        footprints = gpd.GeoDataFrame({
            'geometry': gpd.GeoSeries(geometry(), crs=data['geometry'].crs),
            'centroid': gpd.GeoSeries(centroid(), crs=data['centroid'].crs)
        })

        from validateosm.source.source import Source
        source: Source = next(iter(self._instance.sources.values()))
        identity = source.identify(footprints)
        footprints = footprints.set_index(identity, drop=True)

        footprints['geometry'] = footprints['geometry'].to_crs(3857)
        footprints['centroid'] = footprints['centroid'].to_crs(3857)
        return footprints

    @functools.cached_property
    def _annoy(self) -> AnnoyIndex:
        annoy = AnnoyIndex(2, 'euclidean')
        for i, centroid in enumerate(self.footprints['centroid']):
            annoy.add_item(i, (centroid.x, centroid.y))
        annoy.build(10)
        return annoy

    def __call__(self, data: Union[str, gpd.GeoDataFrame]) -> gpd.GeoDataFrame:
        if isinstance(data, str):
            data: gpd.GeoDataFrame = self._instance.data.xs(data)
        elif not isinstance(data, gpd.GeoDataFrame):
            raise TypeError(type(data))
        data['geometry'] = data['geometry'].to_crs(3857)
        data['centroid'] = data['centroid'].to_crs(3857)
        data['area'] = data['geometry'].area
        footprints = self.footprints

        def footprint() -> Generator[Hashable, None, None]:
            for i, (c, a, g) in enumerate(data[['centroid', 'area', 'geometry']].values):
                for n in self._annoy.get_nns_by_vector((c.x, c.y), 5):
                    match = footprints.iloc[n]
                    geometry: BaseGeometry = match['geometry']
                    if not geometry.intersects(g):
                        continue
                    if geometry.intersection(g).area / a < .5:
                        continue
                    yield match.name
                    break

        index = pd.Series(footprint(), index=data.index, name=footprints.index.name)
        data = data.set_index(index, drop=False, append=True)
        data = data.reorder_levels([index.name, 'name'])
        data = data.sort_index()
        return data
