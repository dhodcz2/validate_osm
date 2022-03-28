from geopandas import GeoSeries, GeoDataFrame
from weakref import WeakKeyDictionary


class FactoryCardinal:
    def __init__(self, compare: object):
        from validate_osm.compare.compare import  Compare
        compare: Compare
        self._compare = compare
        self.data = compare.data.xs('osm', level='name')
        self.aggregate = compare.aggregate.xs('osm', level='name')

    def __enter__(self):
        ...

    def __exit__(self, exc_type, exc_val, exc_tb):
        ...

class DescriptorCardinal:
    def __init__(self):
        self.cache: WeakKeyDictionary[object, GeoDataFrame]

    def __get__(self, instance, owner):
        self._instance = instance
        self._owner = owner
        return self

    def __enter__(self):
        ...

    def __exit__(self, exc_type, exc_val, exc_tb):
        ...

    @property
    def factory(self) -> FactoryCardinal:
        from validate_osm.compare.compare import Compare
        compare: Compare = self._instance
        sources = compare.sources.values()






