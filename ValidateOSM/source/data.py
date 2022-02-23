import abc
import contextlib
import dataclasses
import functools
import inspect
import warnings
from collections import UserDict
from typing import Callable, Any, Iterator, Union, Type
from weakref import WeakKeyDictionary

import pandas
from geopandas import GeoDataFrame
from geopandas import GeoSeries
from pandas import Series
from pandas.core.indexes.range import RangeIndex

from ValidateOSM.source.pipe import DescriptorPipeSerialize
from ValidateOSM.source.scripts import concat


@dataclasses.dataclass
class StructData:
    name: str
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
        return self.func.__qualname__

    @property
    def decorated_func(self):
        def wrapper(source: object):
            obj: object = self.func(source)
            # Expect a GeoSeries
            if self.dtype == 'geometry':
                if self.crs is None:
                    obj = GeoSeries(obj, crs=None)
                else:
                    if not isinstance(obj, GeoSeries):
                        raise TypeError(f"{self.func.__qualname__} must be returned as a GeoSeries so that it may specify a CRS.")
                    if obj.crs is None:
                        raise ValueError(f"{obj} must specify a CRS.")
                    obj = obj.to_crs(self.crs)

            # Got a Series
            if isinstance(obj, Series):
                if not isinstance(obj.index, pandas.core.indexes.range.RangeIndex):
                    warnings.warn(
                        f"{obj}.index returns a {type(obj.index)}; naively passing this may result in a mismatched "
                        f"column index. Resetting this index so that column indices align regardless of implementation."
                    )
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

    class identifier(_DecoratorData):
        """
        In addition to wrapping a function as @data, this marks a particular column as being able to be used as a
        identifier to indicate relationships across datasets.
        """

    class validate(_DecoratorData):
        """
        In addition to wrapping a function as @data, this marks a particular column as one that will be validated as an
        end goal;
        """


class CacheStructs(UserDict):
    """
    Inherits structures from the Source's MRO which will determine the attributes of the Series to be extracted
    in the process of self.raw to self.data
    """

    def __missing__(self, source: type) -> dict[str, StructData]:
        from ValidateOSM.source import Source
        sources: Iterator[Type[Source]] = [
            s for s in source.mro()[::-1]
            if issubclass(s, Source)
        ]
        implications: list[dict[str, StructData]] = [
            self.data[source]
            if source in self.data
            else getattr(source, '_data')
            for source in sources
        ]
        implications.reverse()  # Linear if constructed right-to-left; x^2 if constructed left-to-right
        names: set[str] = {
            name
            for implication in implications
            for name in implication.keys()
        }

        def inherit(name: str) -> StructData:
            try:
                func = source.__dict__[name]
            except KeyError:
                func = None
            else:
                if getattr(func, '__isabstractmethod__', False):
                    func = None
                else:
                    abstract = False

            # Inherit method from mro, ignoring abstract methods
            structs: Iterator[StructData] = (
                imp[name]
                for imp in implications
                if name in imp
            )
            struct = next(structs)
            dtype = struct.dtype
            subtype = struct.subtype
            crs = struct.crs
            dependent = struct.dependent
            if func is not None:
                pass
            elif not struct.abstract:
                func = struct.func
                abstract = True
            else:
                try:
                    func = next(
                        struct.func
                        for struct in structs
                        if not struct.abstract
                    )
                except StopIteration as e:
                    # Abstract methods are not problematic if abc.ABC
                    if isinstance(source, abc.ABC):
                        func = None
                        abstract = True
                    # If not abc.ABC, then only abstract methods were found
                    else:
                        raise NotImplementedError(
                            f"{source.__name__}.{name} inheritance could not be resolved."
                        ) from e
                else:
                    abstract = False
            return StructData(
                name=name,
                func=func,
                dtype=dtype,
                subtype=subtype,
                abstract=abstract,
                crs=crs,
                dependent=dependent
            )

        result = {name: inherit(name) for name in names}
        self.data[source] = result
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
        data = self.data[source] = self.resolve(source)
        return data

    def resolve(self, source: object):
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
            for name, series  in indie.items()
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
        return data

    # def __missing__(self, key):
    #     self._instance = key
    #     self._source = key.__class__
    #     self._structs = self.cache[self._source]
    #     data = self.resolve()
    #     self.data[key] = data
    #     return data
    #
    # @contextlib.contextmanager
    # def subsource(self, subsource: object) -> None:
    #     sources = self._instance.static
    #     self._instance.static = subsource
    #     yield
    #     self._instance.static = sources
    #
    # def resolve(self) -> GeoDataFrame:
    #     from ValidateOSM.source.source import Source
    #     instance: Source = self._instance
    #     self._source: Type[Source]
    #     if isinstance(self._instance.static, Iterator):
    #         sources: Iterator[GeoDataFrame] = (self.resolve_subsource(sub) for sub in instance.static)
    #         data = concat(sources)
    #     else:
    #
    #         data = self.resolve_subsource(instance.static)
    #     data: GeoDataFrame = data.reset_index()
    #     loc = ~data['geometry'].is_valid
    #     data.loc[loc, 'geometry'] = data.loc[loc, 'geometry'].buffer(0)
    #     # data = data.set_geometry('geometry')
    #     return data
    #
    # def resolve_subsource(self, subsource: object) -> GeoDataFrame:
    #     with self.subsource(subsource):
    #         # Extract columns that do not have data dependencies
    #         indie = {
    #             name: struct.decorated_func(self._instance)
    #             for name, struct in self._structs.items()
    #             if not struct.dependent
    #         }
    #         rows = max(len(obj) for obj in indie.values())
    #         data = GeoDataFrame({
    #             name: (series.repeat(rows) if len(series) == 1 else series)
    #             for name, series in indie.items()
    #         })
    #
    #         # Extract data that is dependent on data that has been extracted.
    #         self.data[self._instance] = data
    #         depend: set[StructData] = {
    #             struct
    #             for struct in self._structs.values()
    #             if struct.dependent
    #         }
    #         while depend:
    #             viable: set[StructData] = {
    #                 struct for struct in depend
    #                 if not struct.dependent.difference(data.columns)  # If there is no dif, then all the req are avail
    #             }
    #             if not viable:
    #                 raise RuntimeError(f"Cannot resolve with cross-dependencies: {depend}")
    #             for struct in viable:
    #                 series = struct.decorated_func(self._instance)
    #                 data[struct.name] = series.repeat(rows) if len(series) == 1 else series
    #             depend.difference_update(viable)
    #
    #     return data


class DescriptorData(DescriptorPipeSerialize):
    """
    Allows the user to access the serialized data that has been extracted and processed from the raw data source.
    Source.data.from_file() -> GeoDataFrame
    source().data           -> GeoDataFrame
    """

    _cache = CacheData()

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
