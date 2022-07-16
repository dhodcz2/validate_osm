import abc
import functools
import inspect
import itertools
import logging
from datetime import datetime
from pathlib import Path
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

from validate_osm.source.overpass import DynamicOverpassResource, DescriptorRelations, DescriptorEnumerative
from validate_osm.source.source import Source

from validate_osm.logger import logger, logged_subprocess


# TODO: Apache Airflow scheduling for new OSM entries

class SourceOSM(Source, abc.ABC):
    @property
    def path(self):
        return Path(inspect.getfile(self.__class__)).parent / 'dynamic' / 'source' / self.__class__.__name__ / \
               str( self.bbox)

    def __iter__(self) -> Iterator[GeoDataFrame]:
        yield gpd.read_feather(self.path)

    resource = DynamicOverpassResource()

    def group(self) -> GeoDataFrame:
        data: GeoDataFrame = self.data
        id = [
            element.typeIdShort()
            for element in self.resource
        ]
        data['id'] = id
        self.data = data = data.set_index('id')
        # Set ID as an index

        # append enumerative entries
        logger.debug(f"appending {self.__class__}.resource.enumerative")
        enumerative = self.resource.enumerative
        data = data.append(enumerative)
        # data = data.append(self.resource.enumerative)

        # Group together according to relations
        G = networkx.Graph()
        relations: DescriptorRelations = self.resource.relations
        for group in relations.groups:
            G.add_nodes_from(group)
            G.add_edges_from(zip(group[:-1], group[1:]))
        index = set(data.index)
        groups = connected_components(G)
        # MultiPolygons do not include their ways in the query but nevertheless should be included because their
        #   complex geometry is still returned by OverPass
        multipolygons = {
            element.typeIdShort()
            for element in self.resource
            if element.tag('type') == 'multipolygon'
        }
        incomplete: set[str] = {
            value
            for group in groups
            if not all(id in index for id in group)
               and not any(id in multipolygons for id in group)
            for value in group
        }

        # Drop those who are part of incomplete groups
        data = data.drop(incomplete, errors='ignore')

        complete: list[set[str]] = [
            group for group in groups
            if all(id in index for id in group)
        ]
        data['group'] = pd.Series((
            i
            for i, group in enumerate(complete)
            for _ in group
        ), index=itertools.chain.from_iterable(complete))

        # complete: list[set] = [
        #     group for group in groups
        #     if all(id in index for id in group)
        # ]
        # data['group'] = pd.Series(
        #     data=(
        #         i
        #         for i, group in enumerate(complete)
        #         for _ in group
        #     ), index=itertools.chain.from_iterable(complete)
        # )
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
                    yield mapping[element.gdf()['type']](element.gdf()['coordinates'])
                    continue
                except KeyError as e:
                    raise NotImplementedError(element.gdf()['type']) from e
                except Exception:  # Exception is typically a no-go but .geometry() literally raises this
                    yield None

        gs = GeoSeries(generator(), crs=4326)
        return gs
