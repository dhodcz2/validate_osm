import time
import warnings

warnings.filterwarnings('ignore', '.*area.*')
from shadow.cutil import *
import requests
import concurrent.futures
import io
from typing import Iterator

import numpy as np
import pygeos.creation
from geopandas import GeoDataFrame, GeoSeries
from pandas import Series

import matplotlib.pyplot as plt
import geopandas as gpd


def _get_image(xtile: int, ytile: int, zoom: int, file, max_height: float) -> np.ndarray:
    t = time.time()
    w, n = num2deg(xtile, ytile, zoom, True)
    e, s = num2deg(xtile + 1, ytile + 1, zoom, True)
    dh = (s - n) / 256
    dw = (e - w) / 256

    gcw = np.arange(w, e, dw)
    gce = np.arange(w + dw, e + dw, dw)
    gcn = np.arange(n, s, dh)
    gcs = np.arange(n + dh, s + dh, dh)

    gcnr = np.repeat(gcn, 256)
    gcsr = np.repeat(gcs, 256)
    gcwr = np.tile(gcw, 256)
    gcer = np.tile(gce, 256)
    geometry = pygeos.creation.box(gcwr, gcsr, gcer, gcnr)

    cn = np.repeat(np.arange(256, dtype=np.uint8), 256)
    cw = np.tile(np.arange(256, dtype=np.uint8), 256)

    cells = GeoDataFrame({
        'cn': cn,
        'cw': cw,
    }, geometry=geometry, crs=4326)
    area = abs(dh * dw)

    tile = gpd.read_file(file)[['geometry', 'height']]
    tile['area'] = tile.area
    cells = cells.sjoin(tile)
    cells['weight'] = (
            GeoSeries.intersection(cells.geometry, tile.loc[cells['index_right'], 'geometry'], align=False).area
            / area
            * cells['height']
            / max_height
            * (2 ** 16 - 1)
    )
    weight: Series = cells.groupby(['cn', 'cw'], sort=False, as_index=True).weight.sum()
    weight = weight.astype(np.uint16)
    image = load_image(
        cn=weight.index.get_level_values('cn').values,
        cw=weight.index.get_level_values('cw').values,
        weights=weight.values,
        rows=256,
        columns=256,
    )
    print(f"_get_image takes {(time.time() - t)} seconds")
    return image


def get_images(
        xtiles: list[int], ytiles: list[int], zoom: int, max_height: float
) -> dict[tuple[int, int], np.ndarray]:
    urls = (
        f'http://data.osmbuildings.org/0.2/anonymous/tile/{zoom}/{xtile}/{ytile}.json'
        for xtile, ytile in zip(xtiles, ytiles)
    )

    def func(url, x, y, session: requests.Session):
        response = session.get(url)
        # response = requests.get(url)
        response.raise_for_status()
        return io.StringIO(response.text), x, y

    with concurrent.futures.ThreadPoolExecutor() as pool, requests.Session() as session:
        futures = {
            pool.submit(func, url, xtile, ytile, session)
            for url, xtile, ytile in zip(urls, xtiles, ytiles)
        }
        results: Iterator[tuple[io.StringIO, int, int]] = (
            future.result()
            for future in concurrent.futures.as_completed(futures)
        )
        images = {
            (x, y): _get_image(x, y, zoom, response, max_height)
            for response, x, y in results
        }
    return images


def display_images(
        xtiles: list[int | float], ytiles: list[int | float], zoom: int, max_height: float
) -> None:
    images = get_images(xtiles, ytiles, zoom, max_height)
    for image in images.values():
        plt.figure()
        plt.imshow(image)


def get_image(xtile: int, ytile: int, zoom: int, max_height: float) -> np.ndarray:
    if isinstance(xtile, float):
        xtile, ytile = deg2num(xtile, ytile, zoom, True)
    url = f'http://data.osmbuildings.org/0.2/anonymous/tile/{zoom}/{xtile}/{ytile}.json'
    t = time.time()
    file = io.StringIO(requests.get(url).text)
    print(f"http request takes {(time.time() - t)} seconds")
    return _get_image(xtile, ytile, zoom, file, max_height)


def display_image(xtile: int, ytile: int, zoom: int, max_height: float) -> None:
    image = get_image(xtile, ytile, zoom, max_height)
    plt.figure()
    plt.imshow(image)


class Namespace:
    xtile: list[int]
    ytile: list[int]
    zoom: int
    max: float


if __name__ == '__main__':
    places = [
        (40.740530392418506, -73.99426405495954),
        (40.823631367500454, -73.93789016206937),
        (40.6148254758228, -74.03463693864036),
    ]
    tiles = [
        deg2num(*place[::-1], 15, True)
        for place in places
    ]
    xtiles = [tile[0] for tile in tiles]
    ytiles = [tile[1] for tile in tiles]
    display_images(xtiles, ytiles, 15, 300)
    # %%
    # get_image(*(40.74446651502901, -74.0023459850658)[::-1], 15, 500)
