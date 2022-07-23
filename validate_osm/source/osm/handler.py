import abc
from geopandas import GeoDataFrame, GeoSeries
from pandas import Series, DataFrame

import itertools
import warnings
from abc import abstractmethod
from typing import Iterable, Collection

import networkx
import numpy as np
import osmium

warnings.filterwarnings('ignore', '.*PyGEOS.*')


class MetaOsmiumABC(type(osmium.SimpleHandler), type(abc.ABC), metaclass=type):
    """
    Metaclasses are typically too complicated to be necessary, but the polymorphism of inheriting from both
    osmium.SimpleHandler and RasterStats requires a metaclass that inherits from both of the respective metaclasses,
    Because nothing is implemented here, it inherits __new__ and __init__ directly from osmium.SimpleHandler
    """


class BaseHandler(abc.ABC, osmium.SimpleHandler, metaclass=MetaOsmiumABC):
    @abstractmethod
    def _area(self, a: osmium.osm.Area) -> bool:
        """Qualifier for Area entity"""

    @abstractmethod
    def _relation(self, r: osmium.osm.Relation) -> bool:
        """Qualifier for Relation entity"""

    @abstractmethod
    def _way(self, w: osmium.osm.Way) -> bool:
        """Qualifier for Way entity"""

    def area(self, a: osmium.osm.Area) -> bool:
        if not self.__area(a):
            return False
        if a.from_way():
            id = self._id = f'w{a.orig_id()}'
        else:
            id = self._id = f'r{a.orig_id()}'
        self.geometry[id] = self.wkbfab.create_multipolygon(a.geom())
        return True

    def relation(self, r: osmium.osm.Relation) -> bool:
        if not self.__relation(r):
            return False
        id = self._id = f'r{r.id}'
        members: list[osmium.osm.RelationMember] = r.members
        group = [id]
        group.extend(
            f'{member.type}{member.ref}'
            for member in members
        )
        self._groups.append(group)
        return True

    def __init__(self, *args, **kwargs):
        super(BaseHandler, self).__init__(*args, **kwargs)
        self.geometry: dict[str, str] = {}
        self._groups: list[list[str]] = []
        self._id: str | None = None
        self.wkbfab = osmium.prepared.WKBFactory()

    def apply_file(self, filename, locations=False, idx='flex_mem'):
        super(BaseHandler, self).apply_file(filename, locations, idx)
        self.groups = self.__groups()

    def __groups(self):
        G = networkx.Graph()
        for group in self._groups:
            G.add_nodes_from(group)
            G.add_edges_from(itertools.combinations(group, 2))
        groups: list[Collection[str]] = list(networkx.connected_components(G))
        group = np.arange(len(groups), dtype=np.uint32)
        repeat = np.fromiter((map(len, groups)), dtype=np.uint32, count=len(groups))
        index = np.array(list(itertools.chain.from_iterable(groups)), dtype=object, count=repeat.sum())
        group = np.repeat(group, repeat)

        return Series(group, index=index)
