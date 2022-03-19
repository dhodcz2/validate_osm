from weakref import WeakKeyDictionary
import logging
from geopandas import GeoDataFrame
import time
import abc
import concurrent.futures
import dataclasses
import functools
import inspect
import os
import re
import warnings
from pathlib import Path
from typing import Generator, Any
from typing import Union, Iterable, Optional, Type, Iterator, Collection

import geopandas as gpd
import pandas as pd
import requests
import shapely.geometry
from shapely.geometry import Polygon

from validate_osm.args import global_args as project_args
from validate_osm.util.scripts import concat

logger = logging.getLogger(__name__.partition('.')[0])


@dataclasses.dataclass(repr=False)
class File:
    path: Union[str, Path, None] = dataclasses.field(repr=False)
    url: Optional[str] = dataclasses.field(repr=False, default=None)

    def __post_init__(self):
        self.path = Path(self.path)
        self.name = self.path.name

    def __repr__(self):
        return self.name


def empty_dir(path: Path) -> bool:
    return True if next(path.iterdir(), None) is None else False


#
# def get(self) -> None:
#     logger.info(f'fetching {self}')
#     response = requests.get(self.url, stream=True)
#     response.raise_for_status()
#     if 'Content-Disposition' in response.headers.keys():
#         filename = re.findall('filename=(.+)', response.headers['Content-Disposition'])[0]
#     else:
#         url: str = self.url
#         filename = url.rpartition('/')
#     path: Path = self.path
#     path /= filename
#     with open(path, 'wb') as file:
#         for block in response.iter_content(1024):
#             file.write(block)
#     if self.unzipped:
#         raise NotImplementedError
#         # shutil.unpack_archive()


def get(session: requests.Session, file: File) -> None:
    response = session.get(file.url, stream=True)
    response.raise_for_status()
    logger.debug(f'\t{file.name}')
    logger.debug(f'\t{response.status_code}')
    with open(file.path, 'wb') as f:
        for chunk in response.iter_content(chunk_size=1024):
            if chunk:
                f.write(chunk)
    logger.debug(f'\tdone')


class Resource(abc.ABC):
    name: str
    link: str


class StaticBase(Resource):
    crs: Any
    flipped: bool = False
    project_args = project_args
    preprocess: bool = True

    def crs(self) -> Any:
        ...

    crs = classmethod(property(abc.abstractmethod(crs)))

    def flipped(self) -> bool:
        ...

    flipped = classmethod(property(abc.abstractmethod(flipped)))

    @abc.abstractmethod
    def __get__(self, instance, owner) -> GeoDataFrame:
        ...

    @abc.abstractmethod
    def __delete__(self, instance):
        ...

    @classmethod
    @property
    def directory(cls) -> Path:
        return Path(inspect.getfile(cls)).parent / 'static' / cls.__name__

    @classmethod
    def _from_file(
            cls,
            file: File,
            bbox: Optional[shapely.geometry.Polygon] = None,
            debug=False
            # columns: Optional[list[str]] = None
    ) -> Union[pd.DataFrame, GeoDataFrame]:
        # TODO: Why is file.path different than file.name?
        gdf: Union[GeoDataFrame, pd.DataFrame]
        t = time.time()
        logger.info(f'reading {file.path}')
        match file.path.name.rpartition('.')[2]:
            case 'feather':
                try:
                    gdf = gpd.read_feather(file.path)
                except (AttributeError, TypeError):
                    gdf = pd.read_feather(file.path, )
                if isinstance(gdf, GeoDataFrame) and bbox is not None:
                    gdf = gdf[gdf.within(bbox)]
                if debug:
                    gdf = gdf.head(100)
            case 'parquet':
                try:
                    gdf = gpd.read_parquet(file.path, )
                except (AttributeError, TypeError):
                    gdf = pd.read_parquet(file.path, )
                if isinstance(gdf, GeoDataFrame) and bbox is not None:
                    gdf = gdf[gdf.within(bbox)]
                if debug:
                    gdf = gdf.head(100)
            case _:
                try:
                    gdf = gpd.read_file(file.path, rows=100 if debug else None, bbox=bbox)
                except (AttributeError, TypeError) as e:
                    raise NotImplementedError from e
        logger.info(f'{file.name} took {(time.time() - t) / 60} minutes to load.')
        return gdf

    @classmethod
    def _from_files(
            cls,
            files: list[File],
            bbox: Optional[shapely.geometry.Polygon] = None,
            columns: Optional[list[str]] = None,
            preprocess: bool = True
    ) -> Union[pd.DataFrame, GeoDataFrame]:
        download = [
            file for file in files
            if not file.path.exists()
        ]
        if download:
            # Make all the parent directories
            for file in download:
                if not file.path.parent.exists():
                    os.makedirs(file.path.parent)
            names = ', '.join(file.name for file in download) + '...'
            logger.info(f"fetching {names}")
            with requests.Session() as session, \
                    concurrent.futures.ThreadPoolExecutor() as te:
                future_url_request = [
                    te.submit(get, session, file)
                    for file in download
                ]
                processes = []
                for future in concurrent.futures.as_completed(future_url_request):
                    processes.append(future.result())
                logger.info('done fetching')

        if preprocess:
            preprocess = [
                (file, file.path.parent / (file.path.name.rpartition('.')[0] + '.feather'))
                for file in files
            ]
            preprocessing = [(file, path) for file, path in preprocess if not path.exists()]
            if preprocessing:
                paths = [path for file, path in preprocessing]
                logger.warning(f'Preprecessing {paths}; this may take a while.')
                # TODO: Note: Illinois.geojson.zip is 1.35 GB, but expands in memory to about 5 GB
                for file, path in preprocessing:
                    gdf = cls._from_file(file)
                    t = time.time()
                    logger.info(f'serializing {file.path}')
                    gdf.to_feather(path)
                    logger.info(f'{path.name} to {(time.time() - t) / 60} minutes to serialize.')

            for file, path in preprocess:
                file.path = path

        dfs: Union[Iterator[GeoDataFrame], Iterator[pd.DataFrame]] = (
            cls._from_file(file, bbox, columns)
            for file in files
        )
        if len(files) > 1:
            logger.info(f"concatenating GeoDataFrame from {', '.join(file.name for file in files)}")
            result = concat(dfs)
        else:
            result = next(dfs)
            try:
                result = GeoDataFrame(result)
            except Exception as e:
                raise NotImplementedError(str(e)) from e
            # TODO: If the dataset is in a different crs, where is the most scalable location
            #   to transform that back to 4326?
        return result

    @classmethod
    @abc.abstractmethod
    def from_files(cls) -> GeoDataFrame:
        ...

    def __iter__(self) -> Iterator:
        ...

    @property
    def bbox(self) -> shapely.geometry.Polygon:
        bbox = self._instance.bbox.data
        orientation = (bbox.ellipsoidal if self.flipped else bbox.cartesian)
        gs = gpd.GeoSeries((orientation,), crs=self.crs)
        gs = gs.to_crs(self.crs)
        # I was considering flipping the coords, but maybe just changing ellipsoidal with cartesian is ie
        # gs = gs.map(lambda geom: shapely.ops.transform(lambda x, y, z=None: (y, x, z), geom))

        bbox: shapely.geometry.Polygon = gs.iloc[0]
        return bbox


class StaticNaive(StaticBase):
    """
    Naively load all the files. No caching based on instance
    """
    files: list[File]

    def __init__(
            self,
            files: Union[File, Collection[File]],
            crs: Any,
            name: str,
            link: str,
            flipped=False,
            unzipped=None,
            preprocess: bool = True,
            columns: Union[None, list[str], None] = None
    ):
        if not isinstance(files, File):
            print(f"{files.__class__} != {File}")
        if isinstance(files, File):
            self.files = [files]
        elif isinstance(files, Iterable):
            self.files = list(files)
        else:
            raise TypeError(files)
        self.crs = crs
        self.flipped = flipped
        self.unzipped = unzipped
        self.columns = columns
        self._cache = None
        self.preprocess = preprocess
        self.name = name
        self.link = link

    def __get__(self, instance, owner):
        from validate_osm.source import Source
        self._owner: Source = owner
        self._instance: Type[Source] = instance
        if self._instance is None:
            return self
        if self._cache is None:
            self._cache = self._from_files(self.files, self.bbox)
        return self._cache

    @classmethod
    def from_files(cls) -> GeoDataFrame:
        return cls._from_files(cls.files)

    def __delete__(self, instance):
        self._cache = None


class StaticRegional(StaticBase, abc.ABC):
    class Region(abc.ABC):
        menu: set[str]

        def __contains__(self, item: str):
            return item in self.menu

        def __iter__(self) -> Iterator[str]:
            yield from self.menu

        @abc.abstractmethod
        def __getitem__(self, *items) -> Generator[File, None, None]:
            ...

    @abc.abstractmethod
    def _files_from_polygon(self, item: Polygon) -> Generator[File, None, None]:
        ...

    @property
    @abc.abstractmethod
    def regions(self) -> Iterable['StaticRegional.Region']:
        ...

    @functools.cached_property
    def _menu(self) -> dict[str, Region]:
        return {
            serving: region
            for region in self.regions
            for serving in region
        }

    @functools.cached_property
    def menu(self) -> set[str]:
        return set(self._menu.keys())

    def __getitem__(self, items: Union[str, Polygon, list, tuple]) -> GeoDataFrame:
        if not isinstance(items, tuple):
            items = (items,)

        def gen() -> Iterator[File]:
            for item in items:
                if isinstance(item, str):
                    yield from self._menu[item][item]
                elif isinstance(item, Polygon):
                    yield from self._files_from_polygon(item)
                elif isinstance(item, (tuple, list)):
                    key = item[0]
                    try:
                        yield from self._menu[key][item]
                    except KeyError as e:
                        raise KeyError(
                            f"{self.__class__.__name__}.__getitem__ receieved {items}; did you mean {(list(items),)}?"
                        ) from e
                else:
                    raise TypeError(item)

        files = list(gen())
        return self._from_files(files, bbox=self.bbox)

    def __init__(self):
        self.cache: WeakKeyDictionary[object, GeoDataFrame] = WeakKeyDictionary()

    def __get__(self, instance, owner) -> Union[GeoDataFrame, 'StaticRegional']:
        self._instance = instance
        self._owner = owner
        if instance is None:
            return self
        if instance in self.cache:
            return self.cache[instance]
        else:
            cache = self.cache[instance] = self[self.bbox]
            return cache

    def __delete__(self, instance):
        del self.cache[instance]

    def __contains__(self, item: Union[str, Polygon]):
        match item:
            case str():
                return item in self.menu
            case Polygon():
                try:
                    next(iter(self._files_from_polygon(item)))
                    return True
                except StopIteration:
                    return False
            case _:
                raise TypeError(item)

    def from_files(cls) -> GeoDataFrame:
        raise NotImplementedError
