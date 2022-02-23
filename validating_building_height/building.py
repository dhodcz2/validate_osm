import numpy.typing
from typing import Optional, Iterator, Iterable

import abc
import datetime

import numpy
from geopandas import GeoDataFrame
from pandas import Series

from ValidateOSM.source import data, group, aggregate, Source, SourceOSM, BBox
import dateutil.parser


class SourceBuilding(Source, abc.ABC):
    def identity(self) -> Optional[Series]:
        # TODO: UBID
        return None
        return Series(None, name='ubid', index=self.aggregate.index)

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

    @data('object')
    @abc.abstractmethod
    def address(self):
        """The address of the building"""

    @data('datetime64[ns]')
    @abc.abstractmethod
    def start_date(self):
        """The date at which the building began construction"""


class SourceOSMBuilding(SourceOSM, SourceBuilding, abc.ABC):
    def address(self) -> Iterable[object]:
        housenums = (
            element.tag('addr:housenumber')
            for element in self.source
        )
        streets = (
            element.tag('addr:street')
            for element in self.source
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
            for element in self.source
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
        ids = (ele.id() for ele in self.source)
        buildings = (ele.tag('building') for ele in self.source)
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
