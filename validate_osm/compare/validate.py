from typing import Type, Optional


class DescriptorValidate:
    def __init__(self):
        ...

    def __get__(self, instance, validate):
        from validate_osm.compare.compare import Compare
        self._instance: Optional[Compare] = instance
        self._owner: Type[Compare] = validate
        return self

    def __call__(self, *args, **kwargs):
        # TODO: Perform analytics (in)validating OSM entries
        ...

    def __enter__(self):
        ...

    def __exit__(self, exc_type, exc_val, exc_tb):
        # TODO: Send results to output stream.
        ...
