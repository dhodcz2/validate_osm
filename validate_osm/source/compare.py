from typing import Iterable

if False:
    from .source import Source

class CallableCompare:
    def __call__(self, *sources: tuple[Source]):
        ...

    def __get__(self, instance, owner):
        ...

