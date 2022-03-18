# import dataclasses
# from typing import Callable, Type
# import geopandas as gpd
#
# import pandas as pd
# import inspect
#
# from validate_osm.source.data import StructData
#
#
# @dataclasses.dataclass
# class StructIdentifier:
#     name: str
#     func: Callable
#     dtype: str
#
#     def __hash__(self):
#         return hash(self.name)
#
#     def __eq__(self, other):
#         return self.name == other
#
#     @property
#     def decorated_func(self):
#         def wrapper(source: object):
#             obj: object = self.func(source)
#             if self.dtype == 'geometry':
#                 raise ValueError(
#                     f"{source.__class__.__name__}.{self.func.__name__} cannot use geometry as an identifier")
#             return pd.Series(obj, dtype=self.dtype, name=self.func.__name__)
#
#         return wrapper
#
#
# class DecoratorIdentifier:
#     def __init__(self, dtype):
#         self.dtype = dtype
#
#     args = ['gdf']
#
#     def __call__(self, func: Callable) -> Callable:
#         argspec = inspect.getfullargspec(func)
#         if argspec.args != self.args:
#             raise SyntaxError(
#                 f"{func.__name__} has argspec of {argspec}; {self.__class__.__name__} expects {self.args}"
#             )
#         frame = inspect.currentframe()
#         frames = inspect.getouterframes(frame)
#         frame = frames[1]
#         frame.frame.f_locals.setdefault('_identifier', []).append(
#             StructIdentifier(
#                 name=func.__name__,
#                 func=func,
#                 dtype=self.dtype
#             )
#         )
#
# class DescriptorIdentify:
#     def __get__(self, instance, owner):
#         self._instance = instance
#         self._owner: type = owner
#         return self
#
#     def __call__(self, gdf: gpd.GeoDataFrame) -> pd.Series:
#         from validate_osm.source.source import Source
#         try:
#             # If an identifier has been defined, return its result.
#             struct: StructIdentifier = next(
#                 getattr(source, '_identifier')
#                 for source in self._owner.mro()
#                 if issubclass(source, Source)
#                 and hasattr(source, '_identifier')
#             )
#         except StopIteration:
#             # If an identifier ahs not been defined, just iterate 1,
#             return pd.Series(range(len(gdf)), index=gdf.index, name='i')
#         else:
#             return struct.decorated_func(gdf)
