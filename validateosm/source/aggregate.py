from typing import Callable

from geopandas import GeoDataFrame, GeoSeries

from validateosm.source.groups import Groups


class AggregateFactory:
    def __call__(self, groups: Groups):
        self.groups = groups
        functions = (
            (name, func)
            for cls in self.__class__.mro()
            for name, func in cls.__dict__.items()
            if not name.startswith('_')
               and isinstance(func, Callable)
        )
        result = GeoDataFrame({
            name: func(self)
            for name, func in functions
        })
        result = result.sort_index(axis=0)
        return result

    def geometry(self):
        def data():
            yield from self.groups.ungrouped['geometry'].to_crs(4326)
            yield from (
                gdf['geometry'].to_crs(4326).unary_union
                for gdf in self.groups.grouped
            )

        return GeoSeries(data=data(), index=self.groups.index, crs=4326)

    def centroid(self):
        def data():
            yield from self.groups.ungrouped['centroid'].to_crs(3857)
            yield from (
                gdf['centroid'].to_crs(3857).unary_union.centroid
                for gdf in self.groups.grouped
            )

        return GeoSeries(data=data(), index=self.groups.index, crs=3857)

    def ref(self):
        def data():
            yield from self.groups.ungrouped['ref']
            multi = (
                gdf['ref'].unary_union
                for gdf in self.groups.grouped
            )
            multi = (
                None if union is None
                else union.centroid
                for union in multi
            )
            yield from multi

        return GeoSeries(data=data(), index=self.groups.index)
