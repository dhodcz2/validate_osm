import abc
import concurrent.futures
import dataclasses
import functools
import inspect
import os
import time
from pathlib import Path
from typing import Generator, Any
from typing import Union, Iterable, Optional, Type, Iterator, Collection
from weakref import WeakKeyDictionary

import geopandas as gpd
import pandas as pd
import requests
import shapely.geometry
from geopandas import GeoDataFrame
from shapely.geometry import Polygon

from validate_osm.args import global_args as project_args
from validate_osm.util.scripts import concat, logged_subprocess


@dataclasses.dataclass(repr=False)
class File:
    path: Union[str, Path, None] = dataclasses.field(repr=False)
    url: Optional[str] = dataclasses.field(repr=False, default=None)

    def __post_init__(self):
        self.path = Path(self.path)
        self.name = self.path.name

    def __repr__(self):
        return self.name

    @functools.cached_property
    def preprocessed_path(self) -> Path:
        return self.path.parent / (self.path.name.rpartition('.')[0] + '.feather')


def empty_dir(path: Path) -> bool:
    return True if next(path.iterdir(), None) is None else False


def get(session: requests.Session, file: File) -> None:
    response = session.get(file.url, stream=True)
    response.raise_for_status()
    with open(file.path, 'wb') as f:
        for chunk in response.iter_content(chunk_size=1024):
            if chunk:
                f.write(chunk)


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

    def __get__(self, instance, owner) -> GeoDataFrame:
        self.instance = instance
        self.owner = owner
        return self

    @abc.abstractmethod
    def __delete__(self, instance):
        ...

    @classmethod
    @property
    def directory(cls) -> Path:
        return Path(inspect.getfile(cls)).parent / 'static' / cls.__name__

    def _read_file(
            self,
            path: Path,
            bbox: Optional[shapely.geometry.Polygon] = None,
            debug=False
            # columns: Optional[list[str]] = None
    ) -> Union[pd.DataFrame, GeoDataFrame]:
        from validate_osm.source.source import Source
        self.instance: Source
        gdf: Union[GeoDataFrame, pd.DataFrame]
        with logged_subprocess(self.instance.logger, f'reading {path}'):
            match path.name.rpartition('.')[2]:
                case 'feather':
                    try:
                        gdf = gpd.read_feather(path)
                    except (AttributeError, TypeError):
                        gdf = pd.read_feather(path, )
                    if isinstance(gdf, GeoDataFrame) and bbox is not None:
                        gdf = gdf[gdf.within(bbox)]
                    if debug:
                        gdf = gdf.head(100)
                case 'parquet':
                    try:
                        gdf = gpd.read_parquet(path, )
                    except (AttributeError, TypeError):
                        gdf = pd.read_parquet(path, )
                    if isinstance(gdf, GeoDataFrame) and bbox is not None:
                        gdf = gdf[gdf.within(bbox)]
                    if debug:
                        gdf = gdf.head(100)
                case _:
                    try:
                        gdf = gpd.read_file(path, rows=100 if debug else None, bbox=bbox)
                    except (AttributeError, TypeError) as e:
                        raise NotImplementedError from e
            return gdf

    def _handle_files(
            self,
            files: list[File],
            bbox: Optional[shapely.geometry.Polygon] = None,
            columns: Optional[list[str]] = None,
            preprocess: bool = True
    ) -> Union[pd.DataFrame, GeoDataFrame]:
        from validate_osm.source.source import Source
        self.instance: Optional[Source]
        self.owner: Type[Source]
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
            self.instance.logger.info(f"fetching {names}")
            with requests.Session() as session, \
                    concurrent.futures.ThreadPoolExecutor() as te:
                future_url_request = [
                    te.submit(get, session, file)
                    for file in download
                ]
                processes = []
                for future in concurrent.futures.as_completed(future_url_request):
                    processes.append(future.result())
                self.instance.logger.info('done fetching')

        if preprocess:
            # preprocess = [
            #     (file, file.path.parent / (file.path.name.rpartition('.')[0] + '.feather'))
            #     for file in files
            # ]
            # preprocessing = [(file, path) for file, path in preprocess if not path.exists()]
            # if preprocessing:
            #     paths = [path for file, path in preprocessing]
            #     self.instance.logger.warning(f'Preprecessing {paths}; this may take a while.')
            #     # TODO: Note: Illinois.geojson.zip is 1.35 GB, but expands in memory to about 5 GB
            #     for file, path in preprocessing:
            #         gdf = self._from_file(file)
            #         t = time.time()
            #         self.instance.logger.info(f'serializing {file.path}')
            #         gdf.to_feather(path)
            #         self.instance.logger.info(f'{path.name} to {(time.time() - t) / 60} minutes to serialize.')
            for file in files:
                if not file.preprocessed_path.exists():
                    with logged_subprocess(self.instance.logger, f'Preprocessing {file.preprocessed_path.name}'):
                        gdf = self._read_file(file.path)
                        self.instance.logger.info(f'serializing {file.preprocessed_path.name}')
                        gdf.to_feather(file.preprocessed_path)

        dfs: Union[Iterator[GeoDataFrame], Iterator[pd.DataFrame]] = (
            self._read_file(file.preprocessed_path if preprocess else file.path, bbox, columns)
            for file in files
        )
        if len(files) > 1:
            self.instance.logger.info(f"concatenating GeoDataFrame from {', '.join(file.name for file in files)}")
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
        bbox = self.instance.bbox.data
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
        self.ztpreprocess = preprocess
        self.name = name
        self.link = link

    def __get__(self, instance, owner):
        from validate_osm.source import Source
        self.owner: Source = owner
        self.instance: Type[Source] = instance
        if self.instance is None:
            return self
        if self._cache is None:
            self._cache = self._handle_files(self.files, self.bbox)
        return self._cache

    def from_files(self) -> GeoDataFrame:
        return self._handle_files(self.files)

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
        return self._handle_files(files, bbox=self.bbox)

    def __init__(self):
        self.cache: WeakKeyDictionary[object, GeoDataFrame] = WeakKeyDictionary()

    def __get__(self, instance, owner) -> Union[GeoDataFrame, 'StaticRegional']:
        self.instance = instance
        self.owner = owner
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
