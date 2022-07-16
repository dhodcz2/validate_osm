import dataclasses
import re
from typing import Union, Collection, Any

import pyproj
import shapely.geometry
from shapely.geometry import Polygon


@dataclasses.dataclass
class BBox:
    ellipsoidal: Union[Collection[float], shapely.geometry.Polygon]
    crs: Any = dataclasses.field(default='epsg:4326')
    _ellipsoidal: shapely.geometry.Polygon = dataclasses.field(init=False, repr=False)
    _cartesian: shapely.geometry.Polygon = dataclasses.field(init=False, repr=False)

    @property
    def ellipsoidal(self) -> shapely.geometry.Polygon:
        return self._ellipsoidal

    @ellipsoidal.setter
    def ellipsoidal(self, value):
        if isinstance(value, str):
            string = value.replace(';', ' ')
            string = string.replace(',', ' ')
            string = re.split(r'\s+', string)
            if not len(string) == 4:
                raise ValueError(value)
            value = [float(s) for s in string]
        if isinstance(value, (tuple, list)):
            e = value
            minlat = min(e[0], e[2])
            maxlat = max(e[0], e[2])
            minlong = min(e[1], e[3])
            maxlong = max(e[1], e[3])
            self._ellipsoidal = shapely.geometry.Polygon((
                (minlat, minlong), (maxlat, minlong), (maxlat, maxlong), (minlat, maxlong),
            ))
            self._cartesian = shapely.geometry.Polygon((
                (minlong, minlat), (minlong, maxlat), (maxlong, maxlat), (maxlong, minlat)
            ))
        elif isinstance(value, shapely.geometry.Polygon):
            self._ellipsoidal = value
            self._cartesian = shapely.geometry.Polygon(((y, x) for (x, y) in value.exterior.coords))
        else:
            raise TypeError(value)

    @property
    def cartesian(self) -> shapely.geometry.Polygon:
        return self._cartesian

    def __repr__(self):
        elliposidal = ([
            str(round(bound, 2))
            for bound in self.ellipsoidal.bounds
        ])
        return f"BBox({', '.join(elliposidal)})"

    def __str__(self):
        return '_'.join(
            str(round(bound, 5))
            for bound in self.ellipsoidal.bounds
        )

    def to_crs(self, crs) -> 'BBox':
        if self.crs == crs:
            return self
        trans = pyproj.Transformer.from_crs(self.crs, crs)
        coords = [
            trans.transform(y, x)
            for (y, x)
            in zip(*self.ellipsoidal.exterior.coords.xy)
        ]
        return BBox(shapely.geometry.Polygon(coords), crs=3857)

    def __hash__(self):
        return hash(id(self))

    def __eq__(self, other):
        return id(self) == other
