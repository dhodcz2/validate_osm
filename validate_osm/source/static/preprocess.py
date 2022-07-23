import concurrent
from concurrent.futures import ThreadPoolExecutor
import concurrent.futures
import itertools
import os
from pathlib import Path
from typing import Collection, Iterator

import psutil
import requests

from .file import StructFile, StructFiles, ListFiles
from ..preprocess import CallablePreprocessor

if False:
    from ..source import Source

__all__ = ['StaticPreprocessor']


class StaticPreprocessor(CallablePreprocessor):

    def __new__(cls, *args, **kwargs):
        if not hasattr(cls, '_instance'):
            cls._instance = super().__new__(cls)
        return cls._instance

    def __call__(self, *sources: Collection['Source'], **kwargs):
        files = ListFiles([
            files
            for source in sources
            for files in source.resource[self.bbox]
        ])
        transform = ListFiles([
            file for file in files if
            not file.data.exists()
            or file.source.redo
        ])
        extract = ListFiles([
            file for file in transform
            if not file.raw.exists()
        ])
        self._extract(extract)
        self._load(transform)

    def _download(self, file: StructFile, session: requests.Session):
        url = file.url
        raw = file.raw
        r = requests.get(url, stream=True, session=session)
        parent = raw.parent
        if not parent.exists():
            parent.mkdir(parents=True, exist_ok=True)
        r.raise_for_status()
        with raw.open('wb') as f:
            for chunk in r.iter_content(chunk_size=1024):
                if chunk:
                    f.write(chunk)

    def _extract(self, files: ListFiles):
        file: StructFile | StructFiles
        with ThreadPoolExecutor() as threads, requests.Session() as session:
            threads.map(self._download, files.url, files.raw, itertools.repeat(session))

    def _load(self, files: ListFiles):

        half = psutil.virtual_memory().total // 2
        if psutil.virtual_memory().available < half:
            raise MemoryError('This process requires at least half of the available memory')

        # Transform is sorted from largest to smallest
        files.sort()
        futures = []
        N = len(files)
        threads = concurrent.futures.ThreadPoolExecutor()
        decreasing = len(files)
        increasing = 0

        while (
                decreasing > 0
                or increasing < N
        ):

            # Load files while we can
            while (
                    decreasing > 0
                    and psutil.virtual_memory().available > half
            ):
                decreasing -= 1
                file = files[decreasing]
                futures.append(threads.submit(file.load, file.raw))

            # Consume files while we can't load files
            if (
                    increasing < N
                    and futures[increasing].done()
            ):
                file = files[len(files) - increasing - 1]
                source = file.source
                raw = futures[increasing].result()
                data = file.data
                if not data.exists():
                    data.parent.mkdir(parents=True, exist_ok=True)

                source.resource = raw
                source.data.to_feather(file.data)
                del source.resource
                del source.data
        # TODO: don't forget you still have to work how preprocess() is called among the multiple sources

    @staticmethod
    def _transform(file: StructFile | StructFiles, ):
        source = file.source
        source.resource = file.load(file.raw)
        parent = file.data.parent
        if not parent.exists():
            parent.mkdir(parents=True, exist_ok=True)
        source.data.to_feather(file.data)
        del source.resource
        del source.data
