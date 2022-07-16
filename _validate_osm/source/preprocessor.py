import abc
import concurrent
import logging
import os
from pathlib import Path
from typing import Iterable, Union

import concurrent.futures
import requests

from validate_osm.logger import logger, logged_subprocess

if False | False:
    from validate_osm.source.resource_ import StructFiles, StructFile
    from validate_osm.source.source import Source
    from validate_osm.source.source_osm import SourceOSM


class CallablePreprocessor(abc.ABC):
    # Case: Compare.preprocess(compare.sources)
    @abc.abstractmethod
    def __call__(self, *args: 'Source', **kwargs):
        ...


class CallableStaticPreprocessor(CallablePreprocessor):
    @staticmethod
    def download(url: str, path: Path, session: requests.Session = None) -> None:
        response = session.get(url, stream=True) if session is not None else requests.get(url, stream=True)
        response.raise_for_status()
        with open(path, 'wb') as f:
            for chunk in response.iter_content(chunk_size=1024):
                if chunk:
                    f.write(chunk)

    def extract(self, files: list[Union['StructFile', 'StructFiles']]):
        from validate_osm.source.resource_ import StructFile, StructFiles
        to_download: list[StructFile] = [
            file for file in files
            if issubclass(file.__class__, StructFile)
               and not file.resource.exists()
        ]
        to_download.extend(
            file
            for struct in files
            if issubclass(struct.__class__, StructFiles)
            for file in struct.files
            if not file.resource.exists()
        )
        if to_download:
            for file in to_download:
                if not file.resource.parent.exists():
                    os.makedirs(file.resource.parent)
            logger.info(f'fetching {len(to_download)} files')
            with requests.Session() as session, concurrent.futures.ThreadPoolExecutor() as te:
                future_url_request = [
                    te.submit(self.download, file.url, file.resource, session)
                    for file in to_download
                ]
                processes = []
                for future in concurrent.futures.as_completed(future_url_request):
                    processes.append(future.result())
                logger.debug('done fetching')
        else:
            logger.debug('all files already local')

    def load(self, files: list[Union['StructFiles', 'StructFile']]) -> None:
        with logged_subprocess(f'transforming resources into sources', level=logging.INFO):
            for file in files:
                source: 'Source' = file.source
                path: Path = file.data
                if source.redo or not path.exists():
                    if not path.parent.exists():
                        os.makedirs(path.parent)
                    with logged_subprocess(
                            f'Transforming {source.resource.__class__.__name__} into {source.name}.data', logging.DEBUG
                    ):
                        source.resource = file.load_resource()
                    with logged_subprocess(f'Serializing {source.name}.data', logging.DEBUG):
                        source.data.to_feather(path)
                del source.resource
                del source.data

    def __call__(self, *sources: 'Source', **kwargs):
        files = [
            file
            for source in sources
            for file in source.resource[source.bbox]
        ]
        self.extract(files)
        self.load(files)


class CallableDynamicOverpassPreprocessor(CallablePreprocessor):
    def __call__(self, *sources: 'SourceOSM', **kwargs):
        for source in sources:
            if source.redo or not source.path.exists():
                with logged_subprocess(f'Serializing {source.name}.data', logging.DEBUG):
                    if not source.path.parent.exists():
                        os.makedirs(source.path.parent)
                    source.data.to_feather(source.path)
                del source.resource
                del source.data
