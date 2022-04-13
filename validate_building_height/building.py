import abc
import collections
import datetime
import warnings
from typing import Optional, Iterator, Iterable

import buildingid.code
import dateutil.parser
import geopandas as gpd
import numpy as np
import numpy.typing
import pandas as pd
from geopandas import GeoDataFrame

from validate_osm import (
    DecoratorData,
    Source,
    SourceOSM,
    FactoryAggregate,

)
from validate_osm.source.footprint import CallableFootprint


class BuildingCallableFootprint(CallableFootprint):
    def identify(self, gdf: GeoDataFrame) -> pd.Series:
        def ubid():
            warnings.filterwarnings('ignore', 'geographic CRS')
            # centroid = gdf['centroid'].to_crs(4326)
            geometry: gpd.GeoSeries = gdf['geometry'].to_crs(4326)
            centroid = geometry.centroid
            for x, y, (minx, miny, maxx, maxy) in zip(centroid.x, centroid.y, geometry.bounds.values):
                #     # TODO: codeLength=11 appropriate? Programatically determine code length?
                yield buildingid.code.encode(
                    latitudeLo=miny,
                    latitudeHi=maxy,
                    longitudeLo=minx,
                    longitudeHi=maxx,
                    latitudeCenter=y,
                    longitudeCenter=x,
                    codeLength=11
                )

        result = pd.Series(ubid(), index=gdf.index, name='ubid', dtype='string')
        return result


class BuildingSource(Source):
    footprint = BuildingCallableFootprint

    @staticmethod
    def exclude(self) -> Optional[numpy.typing.NDArray[bool]]:
        # TODO: Exclude things that are too tiny or are clutter

        # TODO: How do we exclude uninteresting or 'garbage' entries? this is originally from SourceOSM

        #     #   Perhpas it is not the duty of containment to determine
        #     # garbage_ways: GeoDataFrame = uncontained[(
        #     #         (uncontained['way'].notna()) &
        #     #         (uncontained['area'] < 20) |
        #     #         (uncontained['way'].isin({
        #     #             way.id()
        #     #             for way in self.source.ways()
        #     #             if way.tag('building') == 'roof'
        #     #         }))
        #     # )]
        #     # garbage_relations: GeoDataFrame = uncontained[(
        #     #         (uncontained['way'].isna()) &
        #     #         (uncontained['area'] < 20) |
        #     #         (uncontained['relation'].isin({
        #     #             relation.id()
        #     #             for relation in self.source.relations()
        #     #             if relation.tag('building') == 'roof'
        #     #         }))
        #     # )]
        #
        #     self.ways['containment'] = pd.Series(np.nan, dtype='Int64')
        #     self.ways['containment'].update(
        #         containers
        #             .loc[containers['way'].notna()]
        #             .set_index('way')
        #             .loc[:, 'containment']
        #     )
        #     self.ways['containment'].update(
        #         contained
        #             .loc[contained['way'].notna()]
        #             .set_index('way')
        #             .loc[:, 'containment']
        #     )
        #     self.relations['containment'] = pd.Series(np.nan, dtype='Int64')
        #     self.relations['containment'].update(
        #         containers
        #             .loc[containers['way'].isna()]
        #             .set_index('relation')
        #             .loc[:, 'containment']
        #     )
        #     self.relations['containment'].update(
        #         contained
        #             .loc[contained['way'].isna()]
        #             .set_index('relation')
        #             .loc[:, 'containment']
        #     )
        #
        return None

    @DecoratorData('string')
    @abc.abstractmethod
    def address(self):
        """The address of the building"""

    @DecoratorData('datetime64[ns]')
    @abc.abstractmethod
    def start_date(self):
        """The date at which the building began construction"""


class BuildingSourceOSM(SourceOSM, BuildingSource, abc.ABC):
    def address(self) -> Iterable[object]:
        housenums = (
            element.tag('addr:housenumber')
            for element in self.resource
        )
        streets = (
            element.tag('addr:street')
            for element in self.resource
        )
        yield from (
            ' '.join((housenum, street))
            if housenum and street
            else None
            for housenum, street in zip(housenums, streets)
        )

    def start_date(self) -> Iterable[datetime.datetime]:
        start_dates: Iterator[Optional[datetime.datetime]] = (
            element.tag('start_date')
            for element in self.resource
        )
        for start_date in start_dates:
            try:
                yield (
                    None if start_date is None
                    else dateutil.parser.parse(start_date)
                )
            except(dateutil.parser.ParserError, TypeError):
                yield None

    @classmethod
    def query(cls, bbox: tuple[int], type='way', appendix: str = 'out meta geom;'):
        if type == 'way':
            return f"""
            (
            way["building"][!"bridge"]["location"!="underground"]{bbox};
            way["building:part"]["location"!="underground"]{bbox};
            );
            """ + appendix
        elif type == 'relation':
            return f"""
            (
            relation["building"]["location"!="underground"]{bbox};
            relation["building:part"]["location"!="underground"]{bbox};
            relation["type"="building"]["location"!="underground"]{bbox};
            );
            """ + appendix
        else:
            raise ValueError(type)

    def containers(self) -> Iterable[bool]:
        ids = (ele.id() for ele in self.resource)
        buildings = (ele.tag('building') for ele in self.resource)
        data: GeoDataFrame = self.data
        if self.ways:
            exclusion = {'roof', 'no', 'bridge', None}
            for id, building in zip(ids, buildings):
                if id not in data.index:
                    yield False
                elif building in exclusion:
                    yield False
                else:
                    yield True
        elif self.relations:
            exclusion = {'roof', 'no', 'bridge', None}
            area = self.data.geometry.to_crs(3857).area
            for id, building in zip(ids, buildings):
                if id not in data.index:
                    yield False
                elif building in exclusion:
                    yield False
                elif area[id] < 40:
                    yield False
                else:
                    yield True
        else:
            raise RuntimeError


class HeightFactoryAggregate(FactoryAggregate):
    def __init__(self, *args, **kwargs):
        super(HeightFactoryAggregate, self).__init__(*args, **kwargs)
        self.index_max_height: dict[int, int] = {}
        self.index_max_floors: dict[int, int] = {}
        self.index_max = collections.ChainMap(self.index_max_height, self.index_max_floors)

    def __call__(self, *args, **kwargs):
        self.index_max_height.clear()
        self.index_max_floors.clear()
        obj = super(HeightFactoryAggregate, self).__call__(*args, **kwargs)
        return obj

    # def height_m(self):
    ##  ignore np.nan
    #     HEIGHT_TO_FLOOR_RATIO = 3.6
    #     data = self._groups.data
    #     iloc = data.loc[data['height_m'].isna() & data['floors'].notna(), 'iloc']
    #     iloc = set(iloc)
    #     data['height_m'] = pd.Series((
    #         floors * HEIGHT_TO_FLOOR_RATIO if i in iloc else height_m
    #         for i, height_m, floors in data[['iloc', 'height_m', 'floors']].values
    #     ), index=data.index, dtype=data['height_m'].dtype
    #     )
    #     index_max_height = self.index_max_height
    #
    #     def gen():
    #         yield from self._groups.ungrouped['height_m']
    #         notna = set(data.loc[data['height_m'].notna(), 'iloc'])
    #         for i, group in enumerate(self._groups.iloc_grouped):
    #             intersection = notna.intersection(group)
    #             if not intersection:
    #                 yield np.nan
    #             else:
    #                 gdf = data.iloc[iter(intersection)]
    #                 valid = zip(gdf['iloc'], gdf['height_m'])
    #                 iloc_max, max = next(valid)
    #                 for loc, height in valid:
    #                     if height > max:
    #                         iloc_max = loc
    #                         max = height
    #                 index_max_height[i] = iloc_max
    #                 yield max
    #
    #     return pd.Series(data=gen(), index=self._groups.index, dtype='Float64')

    def height_m(self):
        # don't ignore np.nan
        HEIGHT_TO_FLOOR_RATIO = 3.6
        data = self._groups.data
        iloc = data.loc[data['height_m'].isna() & data['floors'].notna(), 'iloc']
        iloc = set(iloc)
        data['height_m'] = pd.Series((
            floors * HEIGHT_TO_FLOOR_RATIO
            if i in iloc else height
            for i, height, floors in data[['iloc', 'height_m', 'floors']].values
        ), index=data.index, dtype=data['height_m'].dtype)

        def gen():
            yield from self._groups.ungrouped['height_m']
            index_max_height = self.index_max_height
            for i, gdf in enumerate(self._groups.grouped):
                if any(gdf['height_m'].isna()):
                    yield np.nan
                else:
                    valid = iter(zip(gdf['iloc'], gdf['height_m']))
                    iloc_max, max = next(valid)
                    for loc, height in valid:
                        if height > max:
                            iloc_max = loc
                            max = height
                    index_max_height[i] = iloc_max
                    yield max

        return pd.Series(data=gen(), index=self._groups.index, dtype='Float64')

    # def floors(self):
    #     ignore np.nan
    #     index_max_floors = self.index_max_floors
    #     data = self._groups.data
    #
    #     def gen():
    #         yield from self._groups.ungrouped['floors']
    #         notna = set(data.loc[data['floors'].notna(), 'iloc'])
    #         for i, group in enumerate(self._groups.iloc_grouped):
    #             intersection = notna.intersection(group)
    #             if not intersection:
    #                 yield np.nan
    #             else:
    #                 gdf = data.iloc[iter(intersection)]
    #                 valid = zip(gdf['iloc'], gdf['floors'])
    #                 iloc_max, max = next(valid)
    #                 for loc, height in valid:
    #                     if height > max:
    #                         iloc_max = loc
    #                         max = height
    #                 index_max_floors[i] = iloc_max
    #                 yield max
    #
    #     return pd.Series(data=gen(), index=self._groups.index, dtype='Float64')

    def floors(self):
        def gen():
            yield from self._groups.ungrouped['floors']
            index_max_floors = self.index_max_floors
            for i, gdf in enumerate(self._groups.grouped):
                if any(gdf['floors'].isna()):
                    yield np.nan
                else:
                    valid = iter(zip(gdf['iloc'], gdf['floors']))
                    iloc_max, max = next(valid)
                    for loc, floors in valid:
                        if floors > max:
                            iloc_max = loc
                            max = floors
                    index_max_floors[i] = iloc_max
                    yield max

        return pd.Series(data=gen(), index=self._groups.index, dtype='Float64')

    def timestamp(self):
        def gen():
            yield from self._groups.ungrouped['timestamp']
            series: pd.Series = self._groups.data['timestamp']
            index_max = self.index_max
            yield from (
                series.iat[index_max[i]]
                if i in index_max else np.nan
                for i in range(len(self._groups.iloc_grouped))
            )

        # TODO: look into timezone-aware data
        warnings.filterwarnings('ignore', '.*timezone-aware.*')
        return pd.Series(data=gen(), index=self._groups.index, dtype='datetime64[ns]')

    def start_date(self):
        def gen():
            yield from self._groups.ungrouped['start_date']
            series: pd.Series = self._groups.data['start_date']
            index_max = self.index_max
            yield from (
                series.iat[index_max[i]]
                if i in index_max else np.nan
                for i in range(len(self._groups.iloc_grouped))
            )

        return pd.Series(data=gen(), index=self._groups.index, dtype='datetime64[ns]')

    def address(self):
        def gen():
            yield from self._groups.ungrouped['address']
            series: pd.Series = self._groups.data['address']
            index_max = self.index_max
            yield from (
                series.iat[index_max[i]]
                if i in index_max else np.nan
                for i in range(len(self._groups.iloc_grouped))
            )

        return pd.Series(data=gen(), index=self._groups.index, dtype='string')

    def cardinal(self):
        def gen():
            yield from self._groups.ungrouped.index.get_level_values('id')
            index: pd.MultiIndex = self._groups.data.index.get_level_values('id')
            index_max = self.index_max
            yield from (
                index[index_max[i]]
                if i in index_max else np.nan
                for i in range(len(self._groups.iloc_grouped))
            )

        return pd.Series(data=gen(), index=self._groups.index)


class Height(BuildingSource):
    aggregate_factory = HeightFactoryAggregate

    @DecoratorData('Float64')
    @abc.abstractmethod
    def floors(self):
        """The number of floors in the building"""

    @DecoratorData('Float64')
    @abc.abstractmethod
    def height_m(self):
        """The physical height of the building in meters"""
