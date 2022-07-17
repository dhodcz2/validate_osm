import dataclasses
from shapely.ops import transform
import shapely.ops
import re
from typing import Union, Collection, Any
from shapely.geometry import box

import pyproj
import shapely.geometry
from shapely.geometry import Polygon


@dataclasses.dataclass
class BBox:
    latlon: Polygon
    lonlat: Polygon
    crs: Any

    @classmethod
    def from_latlon(cls, s, w, n, e, crs=4326):
        s, n = min(s, n), max(s, n)
        w, e = min(w, e), max(w, e)
        latlon = box(s, w, n, e)
        longlat = box(w, s, e, n)
        return cls(latlon, longlat, crs)

    @classmethod
    def from_lonlat(cls, w, s, e, n, crs=4326):
        w, e = min(w, e), max(w, e)
        s, n = min(s, n), max(s, n)
        latlon = box(s, w, n, e)
        longlat = box(w, s, e, n)
        return cls(latlon, longlat, crs)

    def to_crs(self, crs: Any):
        project = pyproj.Transformer.from_crs(self.crs, crs).transform
        latlon = transform(project, self.latlon)
        lonlat = transform(project, self.lonlat)
        return BBox(latlon, lonlat, crs)

    def __repr__(self):
        bounds = (
            str(round(bound, 3))
            for bound in self.latlon.bounds
        )
        return f"BBox({', '.join(bounds)})"

    def __str__(self):
        bounds = (
            str(round(bound, 3))
            for bound in self.latlon.bounds
        )
        return f"BBox({', '.join(bounds)})"

    def __hash__(self):
        ...
