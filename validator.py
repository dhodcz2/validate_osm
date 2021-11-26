from validator_util.matches import mismatch_two_addresses
from sources import Source
import functools

from sources import Needles
from collections import namedtuple
from functools import cached_property, partial
from typing import Iterable, NamedTuple, Callable, Union

from geopandas import GeoDataFrame

from sources.bbox import BBox
# from sources.needle import ChicagoNeedles, Needles
# from sources.haystack import ChicagoBuildingFootprints, Haystack
from matcher import Matcher
from validating_height import ChicagoBuildingFootprints, HeightChicagoNeedles


def plot_conflicts(gdf: GeoDataFrame, needles: Needles):
    pass


class Validator:
    def __init__(
        self, needle_fetch: Callable, haystack_fetchers: Union[Callable, Iterable[Callable]],
        bbox: BBox = None
    ):
        """
        :param needle_fetch: Partial() of whatever method is being used to get needle dataframes
        :param haystack_fetchers: Partial() of whatever method is being used to ggget haystack dataframes
        """

        # self.needle_fetchers = [needle_fetch] if isinstance(needle_fetch, Callable) else list(needle_fetch)
        # func = lambda clabl: clabl.func if isinstance(clabl, functools.partial) else clabl
        # needle_src: Source = func(needle_fetch).__self__
        # haystack_fetchers = [haystack_fetchers] if isinstance(haystack_fetchers, Callable) else list(haystack_fetchers)
        #
        # self.identifiers = [*needle_src.identifier_types.keys()]
        # self.bbox = needle_src.regional_bbox_4326 if bbox is None else bbox
        # self.identifiers = [*func(needle_fetch).__self__.identifier_types.keys()]
        # # self.ValidationDataFrames = namedtuple('ValidationDataFrames', (
        # #     f'{func(fetcher).__self__.__qualname__}_{func(fetcher).__name__}'
        # #     for fetcher in haystack_fetchers
        # # ))
        # self.DataFrameTuple = namedtuple('DataFrameTuple', [
        #     f'{func(fetcher).__self__.__name__}_{func(fetcher).__name__}'
        #     for fetcher in haystack_fetchers
        # ])
        #
        # self.needles = needle_fetch(bbox=bbox)
        # self.haystacks = self.DataFrameTuple(*((fetcher(bbox=self.bbox) for fetcher in haystack_fetchers)))
        # self.matcher = Matcher(needles=self.needles)
        #

        self.needle_fetchers = [needle_fetch] if isinstance(needle_fetch, Callable) else list(needle_fetch)
        fetch = lambda thing: thing.func if isinstance(thing, partial) else thing
        needle_source: Needles = fetch(needle_fetch).__self__
        haystack_fetchers = [haystack_fetchers] if isinstance(haystack_fetchers, Callable) else list(haystack_fetchers)

        self.identifiers = [*needle_source.identifier_types.keys()]
        self.bbox = needle_source.regional_bbox_4326_cartesian if bbox is None else bbox
        self.DataFrameTuple = namedtuple('DataFrameTuple', [
            f'{fetch(fetcher).__self__.__name__}_{fetch(fetcher).__name__}'
            for fetcher in haystack_fetchers
        ])
        self.needles = needle_fetch(bbox=bbox)
        self.haystacks = self.DataFrameTuple(
            *(
                fetcher(bbox=self.bbox)
                for fetcher in haystack_fetchers
            )
        )
        # self.haystack = self.DataFrameTuple(
        #     fetcher(bbox=self.bbox)
        #     for fetcher in haystack_fetchers
        # )
        self.matcher = Matcher(needles=self.needles)

    @cached_property
    def compare_
        result = self.DataFrameTuple(*(
            self.matcher.match_by_identifiers(haystack=haystack, identifiers=identifiers)
            for haystack in self.haystacks
        ))
        return result

    @cached_property
    def matches_against_haystacks(self) -> NamedTuple:
        # TODO: With each progressive haystack, do we validate the entirety of needles, or just the unvalidated ones?
        result = self.DataFrameTuple(*(
            self.matcher.match(haystack=haystack, identifiers=self.identifiers)
            for haystack in self.haystacks
        ))
        return result

    @cached_property
    def mismatches_from_distance_algorithm(self) -> NamedTuple:
        matcher = Matcher(needles=self.needles)

        def find_mismatches(candidates: GeoDataFrame):
            mismatches = GeoDataFrame(())
            for ident in self.identifiers:
                ident_hay = ident + '_hay'
                mismatches = mismatches.append(
                    candidates[candidates[ident] != candidates[ident_hay]]
                )
                candidates = candidates[
                    ~candidates.index.isin(mismatches.index)
                ]
            viable_addresses = candidates[pd.notna(candidates['address']) & pd.notna(candidates['address_hay'])]
            mismatches = mismatches.append(
                viable_addresses[
                    viable_addresses['address'].combine(viable_addresses['address_hay'], mismatch_two_addresses)
                ]
            )
            return mismatches

        return self.DataFrameTuple(*(
            find_mismatches(matcher.match_by_distance(haystack=haystack))
            for haystack in self.haystacks
        ))

    @cached_property
    def true_matches(self) -> NamedTuple:
        def find_true_matches(candidates: GeoDataFrame):
            true_matches = GeoDataFrame(())
            for ident in self.identifiers:
                ident_hay = ident + '_hay'
                true_matches = true_matches.append(
                    candidates[candidates[ident] == candidates[ident_hay]]
                )
                candidates = candidates[~candidates.index.isin(true_matches.index)]

            candidates = candidates[  # If address is NA, we cannot confirm a match.
                pd.notna(candidates['address']) & pd.notna(candidates['address_hay'])
                ]
            true_matches = true_matches.append(
                candidates[candidates['address'].combine(candidates['address_hay'], match_two_addresses)]
            )
            return true_matches

        return self.DataFrameTuple(*(
            find_true_matches(gdf) for gdf in self.matches_against_haystacks
        ))

    def plot_conflicts(self, bbox=None, **kwargs):
        pass
        #
        # # TODO: Cached for geom, and cached for center

    # TODO: Add a way on the front-end so that the user may 'disqualify' a row e.g. haystack.row == 0


# from sources.regions import chicago_2000x2000
#
# test_validator = Validator(
#     needle_fetch=HeightChicagoNeedles.from_overpass,
#     haystack_fetchers=partial(
#         ChicagoBuildingFootprints.from_file,
#         filename='sources/haystack_files/Building Footprints (current).zip',
#     ),
#     bbox=chicago_2000x2000
# )
#
if __name__ == '__main__':
    validator = test_validator
    from validator_util.plot import *

    haystack = validator.haystacks[0]
    needles = validator.source
    mismatches = validator.mismatches_from_distance_algorithm[0]
    mismatch = mismatches.sample()
    identifiers = ['bldg_id']
    plot_mismatch(mismatch, needles, haystack, identifiers)
