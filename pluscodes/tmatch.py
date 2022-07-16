import functools
import itertools
from typing import Tuple, Iterator

import numpy as np
from geopandas import GeoDataFrame
from pandas import IndexSlice as idx
from pandas import Series

from pluscodes.util import Decompositions

def _algorithm(left: Series, right: Series) -> dict[str, set[str]]:
    left_tiles = left.index.get_level_values('tile')
    right_tiles = right.index.get_level_values('tile')
    left_groups = left.groupby('space').groups
    """
    Two geospatial datasets, L and R
    
    For each space SL in L
        let JOIN be empty set of spaces
        
        For each tile TL in SL
            If TL in R
                include SR in JOIN where SR contains TL
        
        For each space SR in JOIN
            For each tile TR in SR
                if TR in L and TR not in SL
                    exclude SR from JOIN
        
        join SL and JOIN
    """

    def matches() -> Iterator[Tuple[str, set[str]]]:
        for l_space, l_loc in left_groups.items():
            l_tiles = l_loc.get_level_values('tile')
            l_tiles = l_tiles.intersection(right_tiles)
            r_spaces = right.loc[idx[:, :, l_tiles]].index.get_level_values('space')
            r = right.loc[idx[:, r_spaces, :]]
            r_groups = r.groupby('space').groups

            def r_spaces():
                for r_space, r_loc in r_groups.items():
                    r_tiles = r_loc.get_level_values('tile')
                    for r_tile in r_tiles:
                        if r_tile in left_tiles and r_tile not in l_tiles:
                            break
                    else:
                        yield r_space

            yield l_space, set(r_spaces())

    return {
        l_space: r_spaces
        for l_space, r_spaces in matches()
        if len(r_spaces)
    }


def _match_spaces(left_tiles: Series, right_tiles: Series) -> Series:
    """
    :param left_tiles: The perspective of the join
    :param right_tiles: The dataset that is being matched to the left;
        iloc will match left iloc, while space and tile will remain unchanged
    :return: The index [iloc, space] of the new right DataFrame that matches the left DataFrame on iloc
    """
    dtype = left_tiles.index.get_level_values('space').dtype
    spaces = _algorithm(left_tiles, right_tiles)
    repeat = list(map(len, spaces.values()))
    count = sum(repeat)
    l_spaces = np.fromiter(spaces.keys(), dtype=dtype, count=len(spaces))
    r_spaces = np.fromiter(itertools.chain.from_iterable(spaces.values()), dtype=dtype, count=count)

    iloc = left_tiles.loc[idx[:, l_spaces, :]].index.get_level_values('iloc').unique()
    iloc = iloc.repeat(repeat)
    return Series(iloc, index=r_spaces)

def _tmatch(
        left: Decompositions,
        right: Decompositions,
) -> GeoDataFrame:
    iloc_left = tmatch._match_spaces(left.tiles, right.tiles)
    spaces = right.spaces.droplevel('iloc')
    spaces = spaces.merge(iloc_left, left_on='space', right_index=True, how='right', suffixes=None)
    spaces = spaces.set_index('iloc', append=True)
    return spaces

def tmatch(
        left: GeoDataFrame,
        right: GeoDataFrame,
) -> GeoDataFrame:
    left = Decompositions(left)
    right = Decompositions(right)
    return _tmatch(left, right)



# def match( left: Decompositions, right: Decompositions, ) -> GeoDataFrame:
#     """
#
#     :param left:
#     :param right:
#     :return: a subset of right which matches left on iloc
#     """
#     iloc_left = _match_spaces(left.tiles(), right.tiles())
#     spaces = right.spaces().droplevel('iloc')
#     spaces = spaces.merge(iloc_left, left_on='space', right_index=True, how='right')
#     spaces = spaces.set_index('iloc', append=True)
#     return spaces
#
