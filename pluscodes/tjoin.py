import itertools
from typing import Tuple, Iterator

from geopandas import GeoDataFrame
from pandas import IndexSlice as idx
from pandas import Series

from util import Decompositions


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
            r_spaces = right.loc[idx[:, :, l_tiles], :].index.get_level_values('space')
            r = right.loc[idx[:, r_spaces, :], :]
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


def _iloc_left(left: Series, right: Series) -> Series:
    """
    :param left: The perspective of the join
    :param right: The dataset that is being matched to the left;
        iloc will match left iloc, while space and tile will remain unchanged
    :return: The index [iloc, space] of the new DataFrame that will be appended to left
    """
    spaces = _algorithm(left, right)

    l_spaces = spaces.keys()

    data = left[idx[:, l_spaces, :], :].index.get_level_values('iloc')
    repeat = [
        len(r_spaces)
        for r_spaces in spaces.values()
    ]
    data = data.repeat(repeat)

    r_spaces = itertools.chain.from_iterable(spaces.values())
    index = right.loc[ idx[:, r_spaces, :], : ].index
    index = index.drop(level='tile')

    result = Series(
        data=data,
        index=index,
        name='iloc_left',
    )
    return result

def tjoin(
        left: GeoDataFrame,
        right: GeoDataFrame,
        *args,
        append_index: bool = False,
) -> GeoDataFrame:
    """

    :param left:
    :param right:
    :return: a subset of right which matches left on iloc
    """
    # TODO: This one matches without any residual indices; it leaves no evidence
    left = Decompositions(left)
    right = Decompositions(right)
    iloc_left = _iloc_left(left.tiles(geo=False), right.tiles(geo=False))
    spaces = right.spaces(geo=False)
    spaces = spaces.merge(iloc_left, left_index=True, right_index=True)
    result = left.spaces(geo=False).merge(
        spaces,
        left_on='iloc',
        right_on='iloc_left',
        how='inner',
    )
    return result

