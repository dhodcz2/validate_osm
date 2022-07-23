from collections import UserDict
from ..source.source import Source



class DescriptorSources(UserDict):
    def __getitem__(self, item: str) -> Source:
        ...

    def __init__(self):
        ...



