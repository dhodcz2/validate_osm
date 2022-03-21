from functools import partial as _partial
from typing import Type

from validate_osm.compare.compare import Compare


def partial(*args, **kwargs) -> Type[Compare]:
    """
    This is just a workaround because the PyCharm IDE doesn't seem to accept type hinting an instantiated partial
        as a Type[Compare]
    :param args:
    :param kwargs:
    :return:
    """
    return _partial(*args, **kwargs)
