from geopandas import GeoSeries, GeoDataFrame
from weakref import WeakKeyDictionary


class DescriptorIdentity:
    """
    A: r123, w123, w456
    B: r456, w0123,
    """
    # _dict = WeakKeyDictionary[object, dict] = {}
    # _matrix = WeakKeyDictionary[object, list] = {}

    def __get__(self, instance, owner):
        self._instance = instance
        self._owner = owner
        return self

    @property
    def as_dict(self):
        # if self._instance in self._dict:
        #     return self._dict[self._instance]
        from validate_osm.compare.compare import Compare
        compare: Compare = self._instance

        osm: GeoDataFrame = compare.xs('osm')
        indices = osm.groupby(compare.identity).indices
        ids = osm.index.get_level_values('id')
        result = {
            identity: set(ids[indices])
            for identity, index in indices.items()
        }
        return result


    def dump_csv(self):
        ...

    def dump_json(self):
        ...

