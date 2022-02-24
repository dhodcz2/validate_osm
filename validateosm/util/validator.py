import functools
from typing import Type, Union, Iterable

import pandas as pd
from geopandas import GeoDataFrame

from validateosm.sources import Needles, Haystack
from validateosm.abstracter.apply import Apply
from validateosm.abstracter.combine import Abstracter
from validateosm.util.plot import plot_groupings
from validateosm.compare import Compare


class Validator:
    def __init__(
            self,
            needles: Type[Needles],
            haystacks: Union[list[Type[Haystack]], Type[Haystack]],
            applier: Apply = None,
            bbox=None
    ):
        self.needles = needles
        self.haystacks = haystacks if isinstance(haystacks, list) else [haystacks]
        self.abstracter = Abstracter(needles, applier)

    @property
    def source_needles(self) -> GeoDataFrame:
        return self.needles.from_overpass()

    @property
    def building_needles(self) -> GeoDataFrame:
        return self.abstracter.building_needles

    @functools.cached_property
    def default_bbox(self):
        """A bbox of the southwest corner of the Source, going out .02 degrees lat/lon."""
        bbox = self.building_needles.total_bounds
        bbox = [bbox[0], bbox[1], bbox[0] + 0.02, bbox[1] + 0.02]
        return bbox

    def plot_needles_comparison(self, bbox: tuple[int, int, int, int] = None):
        if bbox is None:
            bbox = self.default_bbox
        source_needles = self.source_needles
        building_needles = self.abstracter.building_needles
        source_needles = source_needles.reset_index().set_index('index')
        source_needles = source_needles.cx[bbox[0]:bbox[2], bbox[1]:bbox[3]]
        building_needles = building_needles.cx[bbox[0]:bbox[2], bbox[1]:bbox[3]]
        source_needles['building_group'] = pd.Series({
            i: building_index
            for source_index, building_index in zip(building_needles['source_index'], building_needles.index)
            for i in source_index
        })
        return plot_groupings(building_needles, source_needles, 'building_group', bbox, flip_ax=True)

    def plot_haystack_comparison(self, haystack: Type[Haystack], bbox: tuple[int, int, int, int] = None):
        if bbox is None:
            bbox = self.default_bbox
        source_haystack = haystack.from_file()
        source_haystack['needles_index'] = self.abstracter.split_on_geometry(source_haystack)
        building_haystack = self.abstracter.abstractify(haystack)
        source_haystack = source_haystack.cx[bbox[0]:bbox[2], bbox[1]:bbox[3]]
        building_haystack = building_haystack.cx[bbox[0]:bbox[2], bbox[1]:bbox[3]]
        source_haystack['building_group'] = pd.Series({
            i: building_index
            for source_index, building_index in zip(building_haystack['source_index'], building_haystack.index)
            for i in source_index
        })
        # return plot_groupings(building_haystack, source_haystack, 'building_group', bbox, flip_ax=True)
        return plot_groupings(
            building_haystack,
            source_haystack,
            'building_group',
            bbox=bbox,
            flip_ax=True
        )

    def plot_needles_haystack_comparison(self, haystack: Type[Haystack], bbox: tuple[int, int, int, int] = None):
        if bbox is None:
            bbox = self.default_bbox
        building_needles = self.abstracter.building_needles
        building_haystack = self.abstracter.abstractify(haystack)
        # building_needles = building_needles.cx[bbox[0]:bbox[2], bbox[1]:bbox[3]]
        # building_haystack = building_haystack.cx[bbox[0]:bbox[2], bbox[1]:bbox[3]]
        return plot_groupings(
            building_needles,
            building_haystack,
            'needles_index',
            bbox=bbox,
            plot_hatches=(True, True),
        )

    # TODO: Perhaps just store bbox in the instance instead of passing it around everywhere.
    def plot_needles_haystacks_discrepancies(self, haystack: Type[Haystack], bbox: tuple[int, int, int, int] = None):
        # TODO: Double check which side(s) get hatches
        building_needles = self.abstracter.building_needles
        source_haystack = haystack.from_file()
        source_haystack['needles_index'] = self.abstracter.split_on_geometry(source_haystack)
        building_haystack = self.abstracter.abstractify(source_haystack)
        # matches = building_needles.merge(building_haystack, how='inner', left_index=True, right_on='needles_index',)
        building_haystack = building_haystack[building_haystack['needles_index'].notna()]
        matches = building_needles.merge(
            building_haystack,
            how='inner',
            left_index=True,
            right_on='needles_index',
            suffixes=[None, '_r'],
        )
        matches = matches[matches['height_m'].notna() & matches['height_m_r'].notna()]
        matches['percent_error'] = pd.Series((
            (hr - hl) / hr
            if hr > hl else
            (hl - hr) / hl
            for hl, hr in zip(matches['height_m'], matches['height_m_r'])
        ), index=matches.index)

        print(matches['percent_error'].median())
        from validateosm.util.plot import plot_discrepancies
        return plot_discrepancies(matches, 'percent_error')




class Validator:
    def __init__(self, needles: Type[Needles]):
        self.needles = needles
        self.abstraction_needles: GeoDataFrame = needles.from_abstraction()
        self.source_needles: GeoDataFrame = needles.from_source()

    def match(self, haystack: Type[Haystack]):
        abstraction_haystack: GeoDataFrame = haystack.from_abstraction()
        if 'needles_index' not in abstraction_haystack:
            raise ValueError(f'needles_index must be a column in {haystack}.from_abstraction()')
        match = self.abstraction_needles.merge(
            abstraction_haystack,
            left_index=True,
            right_on='needles_index',
            how='inner',
            suffixes=[None, '_h']
        )
        for validating in self.needles.validating_types.keys():
            loc = match[validating].notna() & match[validating + '_h']
            match.loc[loc, 'error_' + validating] = pd.Series((
                (vr - vl) / vr
                if vr > vl else
                (vl - vr) / vl
                for vl, vr in zip(match[validating], match[validating + '_h'])
            ), index=match.loc[loc].index)
        return match


class Validator:
    def __init__(self, needles: Type[Needles], haystacks: Union[Type[Haystack], Iterable[Type[Haystack]]]):
        self.needles = needles.from_abstraction()
        self.name = needles.name
        # self.compare = Compare([osm, *haystacks if isinstance(haystacks, list) else haystacks])
        self.compare = Compare(
            haystacks if isinstance(haystacks, list) else [haystacks] + [needles]
        )


