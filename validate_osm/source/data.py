import dataclasses
import inspect
from collections import UserDict
from typing import Callable, Any, Iterator, Union
from weakref import WeakKeyDictionary

import pandas
import pandas as pd
from geopandas import GeoDataFrame
from geopandas import GeoSeries
from pandas import Series
from pandas.core.indexes.range import RangeIndex

from validate_osm.source.pipe import DescriptorPipeSerialize


# TODO: Replace DescriptorData with __enter__, __exit__, __get__, and __set__ from compare.data instead of messing
#   around with CacheData, CacheStruct, etc.

@dataclasses.dataclass
class StructData:
    name: str
    cls: str
    func: Callable
    dtype: str
    subtype: type
    abstract: bool
    crs: Any
    dependent: set[str]

    def __hash__(self):
        return hash(self.name)

    def __eq__(self, other):
        return self.name == other

    def __repr__(self):
        return f"{self.__class__.__name__}[{self.name} from {self.cls}]"

    @property
    def decorated_func(self):
        def wrapper(source: object):
            if self.abstract:
                raise TypeError(f"{self.__repr__()} is abstract method.")
            obj: object = self.func(source)
            # Expect a GeoSeries
            if self.dtype == 'geometry':
                if self.crs is None:
                    obj = GeoSeries(obj, crs=None)
                else:
                    if not isinstance(obj, GeoSeries):
                        raise TypeError(
                            f"{self.func.__qualname__} must be returned as a GeoSeries so that it may specify a CRS."
                        )
                    if obj.crs is None:
                        raise ValueError(f"{obj} must specify a CRS.")
                    obj = obj.to_crs(self.crs)

            # Got a Series
            if isinstance(obj, Series):
                if not isinstance(obj.index, pandas.core.indexes.range.RangeIndex):
                    # source.logger.debug(
                    #     f"index returns a {type(obj.index)}; naively passing this may result in a mismatched "
                    #     f"column index. Resetting this index so that column indices align regardless of implementation."
                    # )
                    obj = obj.reset_index(drop=True)
                if self.dtype is not None:
                    obj = obj.astype(self.dtype)
            else:
                obj = Series(obj, dtype=self.dtype)
            return obj

        return wrapper


class _DecoratorData:
    def __init__(self, dtype, crs=None, dependent: Union[str, set[str]] = set()):
        self.dtype = dtype
        self.crs = crs
        self.dependent = {dependent} if isinstance(dependent, str) else set(dependent)

    args = ['self']

    def __call__(self, func: Callable) -> Callable:
        argspec = inspect.getfullargspec(func)
        if argspec.args != self.args:
            raise SyntaxError(
                f"{func.__name__} has argspec of {argspec}; {self.__class__.__name__} expects {self.args}"
            )
        frame = inspect.currentframe()
        frames = inspect.getouterframes(frame)
        frame = frames[1]
        _data: dict[str, StructData] = frame.frame.f_locals.setdefault('_data', {})
        name = func.__name__
        _data[name] = StructData(
            name=func.__name__,
            func=func,
            cls=frame.frame.f_locals['__qualname__'],
            dtype=self.dtype,
            subtype=self.__class__,
            abstract=getattr(func, '__isabstractmethod__', False),
            crs=self.crs,
            dependent=self.dependent
        )
        return func


class DecoratorData(_DecoratorData):
    """
    Wraps a function as @data to be extracted into a Series that is standardized and uniform across different
        datasets.
    """

    # class identifier(_DecoratorData):
    #     """
    #     In addition to wrapping a function as @data, this marks a particular column as being able to be used as a
    #     identifier to indicate relationships across datasets.
    #     """

    class validate(_DecoratorData):
        """
        In addition to wrapping a function as @data, this marks a particular column as one that will be validated as an
        end goal;
        """


class CacheStructs(UserDict):
    data: dict[type, dict[str, StructData]]

    def __missing__(self, source: type) -> dict[str, StructData]:
        from validate_osm.source.source import Source
        sources = [
            s for s in source.mro()[:0:-1]
            if issubclass(s, Source)
        ]
        # start reversed to maintain O(x) instead of O(x^2)
        inheritances = [
            self.__getitem__(source)
            # self.data[source]
            for source in sources
        ]
        inheritances.reverse()  # undo reverse; we check the top inheritances first
        names: set[str] = set(getattr(source, '_data', {}).keys())
        names.update(
            key for inherit in inheritances
            for key in inherit.keys()
        )

        def inherit(name: str) -> StructData:
            # Base case: the class has decorated a function with @data

            if hasattr(source, '_data') and name in (structs := getattr(source, '_data', {})):
                return structs[name]

            struct_list: Iterator[StructData] = [
                inherit[name]
                for inherit in inheritances
                if name in inherit
            ]
            structs = iter(struct_list)
            try:
                struct = next(structs)
            except StopIteration as e:
                raise RuntimeError from e

            # Get first inheritance
            dtype = struct.dtype
            subtype = struct.subtype
            crs = struct.crs
            dependent = struct.dependent

            if name in source.__dict__:
                func = getattr(source, name)
                abstract = getattr(func, '__isabstractmethod__', False)
                if abstract:
                    raise RuntimeError(f"{source=}; {name=}, {abstract=}")
                cls = source.__name__
            else:
                # Get first inheritance that isn't abstract
                if struct.abstract:
                    for struct in structs:
                        if not struct.abstract:
                            break
                abstract = struct.abstract
                func = struct.func
                cls = struct.cls

            return StructData(
                name=name,
                func=func,
                dtype=dtype,
                subtype=subtype,
                abstract=abstract,
                crs=crs,
                cls=cls,
                dependent=dependent
            )

        result = self.data[source] = {name: inherit(name) for name in names}
        return result


class CacheData(UserDict):
    """
        Handles the extraction of standardized, interchangeable data from the raw data source set to the attribute of
        self.raw within Source.__init__
    """
    structs: CacheStructs[type, dict[str, StructData]] = CacheStructs()

    def __init__(self):
        super(CacheData, self).__init__()
        self.data: WeakKeyDictionary[object, GeoDataFrame] = WeakKeyDictionary()

    def __missing__(self, source: object) -> GeoDataFrame:
        from validate_osm.source.source import Source
        source: Source

        data = self.data[source] = self._data(source)
        if 'group' not in self.data[source].index.names:
            raise ValueError(f"{source.group.__qualname__} has not assigned a group index")
        source.logger.debug(f'{source.__class__.__name__}.data done; deleting {source.resource.__class__.__name__}')
        del source.resource
        return data

    def __setitem__(self, key, gdf: GeoDataFrame):
        gdf['iloc'] = pd.Series(range(len(gdf)), dtype='int32', index=gdf.index)
        self.data[key] = gdf

    def _data(self, source: object):
        from validate_osm.source.source import Source
        source: Source
        structs = self.structs[source.__class__]
        indie = {
            name: struct.decorated_func(source)
            for name, struct in structs.items()
            if not struct.dependent
        }
        rows = max(len(obj) for obj in indie.values())
        data = GeoDataFrame({
            name: (
                series.repeat(rows).reset_index(drop=True)
                if len(series) == 1
                else series
            )
            for name, series in indie.items()
        })

        # Extract data that is dependent on data that has been extracted.
        self.data[source] = data
        depend: set[StructData] = {
            struct
            for struct in structs.values()
            if struct.dependent
        }
        while depend:
            viable: set[StructData] = {
                struct for struct in depend
                if not struct.dependent.difference(data.columns)  # If there is no dif, then all the req are avail
            }
            if not viable:
                raise RuntimeError(f"Cannot resolve with cross-dependencies: {depend}")
            for struct in viable:
                series = struct.decorated_func(source)
                data[struct.name] = (
                    series.repeat(rows).reset_index(drop=True)
                    if len(series) == 1
                    else series
                )
                # data[struct.name] = series.repeat(rows) if len(series) == 1 else series
            depend.difference_update(viable)
        data = source.group()
        return data


class DescriptorData(DescriptorPipeSerialize):
    """
    Allows the user to access the serialized data that has been extracted and processed from the raw data source.
    Source.data.from_file() -> GeoDataFrame
    source().data           -> GeoDataFrame
    """

    _cache = CacheData()

    __get__: Union[GeoDataFrame, 'DescriptorData']
    name = 'data'

    @property
    def validating(self) -> set[str]:
        structs = self._cache.structs[self._owner]
        return {
            name
            for name, struct in structs.items()
            if struct.subtype is DecoratorData.validate
        }

    @property
    def identifier(self) -> set[str]:
        structs = self._cache.structs[self._owner]
        return {
            name
            for name, struct in structs.items()
            if struct.subtype is DecoratorData.identifier
        }

    @property
    def structs(self) -> dict[str, StructData]:
        return self._cache.structs[self._owner]
