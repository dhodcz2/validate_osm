import abc
import functools
import itertools
import logging
from datetime import datetime
from typing import Iterable, Hashable
from typing import Iterator

import geopandas as gpd
import networkx
import pandas as pd
import shapely
from geopandas import GeoDataFrame, GeoSeries
from networkx import connected_components
from pandas import Series
from shapely.geometry import Polygon

from validate_osm.source.overpass import DynamicOverpassResource, DescriptorRelations
from validate_osm.source.source import Source

logger = logging.getLogger(__name__.partition('.')[0])

# TODO: Apache Airflow scheduling for new OSM entries

class SourceOSM(Source, abc.ABC):
    resource = DynamicOverpassResource()

    def group(self) -> GeoDataFrame:
        data: GeoDataFrame = self.data
        data['id'] = self.id
        self.data = data = data.set_index('id')
        # Set ID as an index

        # append enumerative entries
        logger.debug(f"appending {self.__class__}.resource.enumerative")
        data = data.append(self.resource.enumerative)

        # Group together according to relations
        G = networkx.Graph()
        # for group in self.resource.relations._groups:
        # sel
        relations: DescriptorRelations = self.resource.relations
        for group in relations.groups:
            G.add_nodes_from(group)
            G.add_edges_from(zip(group[:-1], group[1:]))
        index = set(data.index)
        groups = connected_components(G)
        incomplete: Iterable[Hashable] = [
            value
            for group in groups
            if not all(id in index for id in group)
            for value in group
        ]

        # Drop those who are part of incomplete groups
        data = data.drop(incomplete, errors='ignore')
        complete: list[set] = [
            group for group in groups
            if all(id in index for id in group)
        ]
        data['group'] = pd.Series(
            data=(
                i
                for i, group in enumerate(complete)
                for member in group
            ), index=itertools.chain.from_iterable(complete)
        )
        data: gpd.GeoDataFrame = data.set_index('group', append=True)
        # Append enumerative groups
        return data

    @classmethod
    @abc.abstractmethod
    def query(cls, bbox: tuple[int], type='way', appendix: str = 'out meta geom;'):
        ...

    @functools.cached_property
    def id(self) -> Iterator[str]:
        return [
            element.typeIdShort()
            for element in self.resource
        ]

    def timestamp(self) -> Iterable[datetime]:
        generator: Iterator[datetime] = (
            # datetime.strptime(element.timestamp(), '%Y-%m-%dT%H:%M:%SZ')
            element.timestamp()
            for element in self.resource
        )
        return Series(generator, dtype='datetime64[ns]')

    def geometry(self) -> GeoSeries:
        mapping = {
            'Polygon': lambda coordinates: Polygon(coordinates[0]),
            'LineString': lambda coordinates: None,
            'MultiPolygon': lambda coordinates: shapely.geometry.MultiPolygon((
                Polygon(polygon[0])
                for polygon in coordinates
            ))
        }

        def generator():
            for element in self.resource:
                if not (element.type() == 'way' or element.tag('type') == 'multipolygon'):
                    yield None
                    continue
                try:
                    yield mapping[element.geometry()['type']](element.geometry()['coordinates'])
                    continue
                except KeyError as e:
                    raise NotImplementedError(element.geometry()['type']) from e
                except Exception:  # Exception is typically a no-go but .geometry() literally raises this
                    yield None

        gs = GeoSeries(generator(), crs=4326)
        return gs
