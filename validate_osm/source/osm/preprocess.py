from ..static.preprocess import StaticPreprocessor
import concurrent
from concurrent.futures import ThreadPoolExecutor
import concurrent.futures
import itertools
import os
from pathlib import Path
from typing import Collection, Iterator, Type

import psutil
import requests

from ..static.file import StructFile, StructFiles, ListFiles
from ..preprocess import CallablePreprocessor

if False:
    from ..source import Source

from ..static.file import StructFile, ListFiles, StructFiles
from .handler import BaseHandler
from .resource_ import ResourceOsmium, DescriptorResource


class OSMPreprocessor(StaticPreprocessor):

    def __load(self, files: ListFiles):
        half = psutil.virtual_memory().total // 2
        if psutil.virtual_memory().available < half:
            raise MemoryError('This process requires at least half of the available memory')

        # Transform is sorted from largest to smallest
        files.sort()
        file: StructFile | StructFiles
        for file in files:
            source: 'Source' = file.source
            resource: 'ResourceOsmium' = source.resource
            Handler: Type['BaseHandler'] = resource.Handler
            handler = Handler()
            handler.apply_file(file)

            source.resource = handler
            source.data.to_feather(file.data)
            del source.resource
            del source.data
