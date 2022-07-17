import concurrent
import concurrent.futures
import itertools
import os
from pathlib import Path
from typing import Collection

import psutil
import requests

from .file import StructFile, StructFiles, ListFiles
from ..preprocess import CallablePreprocessor


class StaticPreprocessor(CallablePreprocessor):
    # def __call__(self, *sources, **kwargs):
    #     files: list[StructFile, StructFiles] = [
    #         file
    #         for source in sources
    #         for file in source.resource[self.bbox]
    #     ]
    #     extract = [
    #         file for file in files
    #         if (
    #                 not file.data.exists()
    #                 or file.source.redo
    #                 and not file.raw.exists()
    #         )
    #     ]
    #     self.__extract(extract)
    #     transform = [
    #         file for file in extract
    #         if (
    #                 not file.data.exists()
    #                 or file.source.redo
    #         )
    #     ]
    #     with concurrent.futures.ThreadPoolExecutor() as threads:
    #         paths = (file.data.parent for file in files if not file.data.parent.exists())
    #         threads.map(paths, os.makedirs)
    #     self.__load_transform(transform)


    def __preprocess(self, *sources: Collection['Source'], **kwargs):
        files = ListFiles([
            files
            for source in sources
            for files in source.resource[self.bbox]
        ])
        extract = ListFiles([
            file for file in files if
            not file.data_.exists()
            or file.source.redo
            and not file.raw.exists()
        ])
        self.__extract(extract)
        transform = ListFiles([
            file for file in extract if
            not file.data_.exists()
            or file.source.redo
        ])
        with concurrent.futures.ThreadPoolExecutor() as threads:
            paths = (file.data_.parent for file in files if not file.data_.parent.exists())
            threads.map(paths, Path.mkdir)
        self.__load_transform(transform)


    def __extract(self, files: ListFiles):
        with requests.Session() as session, concurrent.futures.ThreadPoolExecutor() as threads:
            paths = (file.raw.parent for file in files if not file.raw.parent.exists())
            threads.map(paths, os.makedirs)
            url = (file.url for file in files)
            raw = (file.raw for file in files)
            session = itertools.repeat(session)
            threads.map(self.download, url, raw, session)

    def __load_transform(self, files: ListFiles):
        # TODO: Perhaps there is a way we can exploit parallelism when performing the transform
        #   however I cannot think of a way that is safe for memory consumption
        #   regardless I don't think this step is too time consuming
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
                load = file.load
                futures.append(threads.submit(load, file.raw))

            # Consume files while we can't load files
            if (
                    increasing < N
                    and futures[increasing].done()
            ):
                file = files[len(files) - increasing - 1]
                source = file.source
                raw = futures[increasing].result()
                increasing += 1

                # These two lines are literally what start the transform process
                source.resource = raw
                source.data_.to_feather(file.data_)

                del source.resource
                del source.data_

        # TODO: don't forget you still have to work how preprocess() is called among the multiple sources

    @staticmethod
    def __transform(file: StructFile | StructFiles, ):
        source = file.source()
        source.resource = file.load(file.raw)
        source.data.to_feather(file.data)

    @staticmethod
    def download(url: str, path: Path, session: requests.Session = None) -> None:
        response = session.get(url, stream=True) if session is not None else requests.get(url, stream=True)
        response.raise_for_status()
        with open(path, 'wb') as f:
            for chunk in response.iter_content(chunk_size=1024):
                if chunk:
                    f.write(chunk)
