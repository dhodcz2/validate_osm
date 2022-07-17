from typing import Collection
from collections import defaultdict
import abc

# noinspection PyUnreachableCode
if False:
    from .source import Source


class CallablePreprocessor(abc.ABC):
    """
    Preprocessors are Singletons that take a set of Sources, and provide ETL ops for Source.data,
    leveraging multithreading and or multiprocessing.
    """

    def __new__(cls, *args, **kwargs):
        if not hasattr(cls, '_instance'):
            cls._instance = super().__new__(cls, *args, **kwargs)
        return cls._instance

    def __get__(self, instance, owner):
        self.resource = instance
        return self

    def __call__(self, *sources: 'Source', **kwargs):
        preprocesses: defaultdict[CallablePreprocessor, list[Source]] = defaultdict(list)
        for source in sources:
            preprocesses[source.preprocess].append(source)
        for preprocess, sources in preprocesses.items():
            preprocess.__preprocess(sources, **kwargs)

    @abc.abstractmethod
    def __preprocess(self, *sources: Collection['Source'], **kwargs):
        ...


"""
preprocess(bbox, *sources)

"""

"""
# sources = []
# preprocesses = {source.preprocess for source in sources}
# for preprocess in preprocesses:
#     preprocess(sources)


sources = [TypeSource]
preprocesses = {source.resource.preprocess(self.bbox) for source in sources}

with concurrent.futures.ThreadPoolExecutor() as threads:
    
"""
