from ctypes import Union
from typing import Iterator
from weakref import WeakKeyDictionary

class DescriptorRaw:
    # TODO: Allow the programmer to entire a URL to be downloaded within self.__init__
    #   The file downloaded is then used as raw data to be extracted, grouped, and aggregated
    # TODO: Develop an interface that facilitates easy sharing of extracted data across programmers involved
    #   So that each programmer need not download the original source; they can get the .aggregate and .data from a
    #   collaborator and save the hassle.
    _cache_raw: WeakKeyDictionary[object, Union[object, Iterator[object]]] = WeakKeyDictionary()

    def __init__(self):
        ...

    def __get__(self, instance, owner):
        self.instance = instance
        self.owner = owner
        return self


