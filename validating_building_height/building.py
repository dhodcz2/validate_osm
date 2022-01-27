from typing import Optional, Iterator, Iterable

import abc
import datetime

from geopandas import GeoDataFrame

from ValidateOSM.source import data, group, aggregate, Source, SourceOSM, BBox
import dateutil.parser

class SourceBuilding(Source, abc.ABC):
    @data('object')
    @abc.abstractmethod
    def address(self) -> Iterable[object]:
        """The address of the building"""

    @data('datetime64[ns]')
    @abc.abstractmethod
    def start_date(self) -> Iterable[datetime.datetime]:
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
