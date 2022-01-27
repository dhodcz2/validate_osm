from pandas import DataFrame
from functools import reduce

from typing import Type, Iterator

import pandas as pd
from pandas import IndexSlice as idx

from ValidateOSM.sources import Source, Haystack, Needles


# TODO: Rename name to abbreviation
# TODO: Make sure all have the same 'abstraction' index name
class Compare:
    def __init__(self, sources: list[Type[Source]]):
        self.sources: dict[str, Source] = {
            source.name: source
            for source in sources
        }
        self.validating: set[str] = {
            validating
            for source in sources
            for validating in source.validating_types.keys()
        }
        self.comparison: DataFrame = pd.concat((
            source.from_abstraction()[self.validating]
                .assign(n=source.name)
                .set_index('n', append=True)
            for source in sources
        ))
        # TODO:
        names: list[str] = [source.name for source in sources]
        self.memo: dict[str, dict[str, DataFrame]] = {
            n1: {
                n2: {
                    pd.Series(0, index=self.comparison.xs(n2, level='n').index)
                    if n1 == n2 else None
                }
                for n2 in names
            }
            for n1 in names
        }

    def oned(self, left: str, right: str) -> DataFrame:
        # Index(index) -> validating
        # df.loc[i] = #
        """
        index: { height, floors }
        """
        # Determine the indices at which both exist
        if (result := self.memo[right][left]) is not None:
            return result

        loc = pd.Index.union(
            self.comparison.xs(left, level='name').index,
            self.comparison.xs(right, level='name').index
        )
        left: DataFrame = self.comparison.loc[idx[loc, left]]
        right: DataFrame = self.comparison.loc[idx[loc, right]]

        result: DataFrame = DataFrame({
            v: {
                (r - l) / r
                if r > l else
                (l - r) / l
                for l, r in zip(left[v], right[v])
            }
            for v in self.validating
        }, index=loc)
        return result

    def twod(self, defendant: str, jury: list[str]) -> DataFrame:
        # MultiIndex(index, validating) -> [name]
        # df.loc[idx[i, v], n] = #
        """
        index: {
            height: { lod, 3dm }
            floors: { lod, 3dm }
        }
        """
        result = pd.concat((
            self.oned(defendant, j)
                .unstack()
                .swaplevel(0, 1)
                .rename(j)
            for j in jury
        ), axis=1)
        result = result.sort_index()
        return result

    def threed(self, peers: list[str]) -> list[DataFrame]:
        # MultiIndex(index, validating, name) -> [name]
        # df.loc[idx[i, v, n]] = #
        """
        index:  {
            height: {
                lod: {lod, 3dm, osm}
                3dm: {lod, 3dm, osm}
                osm: {lod, 3dm, osm}
            },
            floors: {
                lod: {lod, 3dm, osm}
                3dm: {lod, 3dm, osm}
                osm: {lod, 3dm, osm}
            }
        }
        """
        # TODO: This is currently wasteful because less than half as calculations must be made
        #   if defendant == peer, 0
        #   if defendant->peer, peer->defendant is free
        result = pd.concat((
            self.twod(defendant, peers)
                .assign(n=defendant)
                .set_index('n', append=True)
            for defendant in peers
        )).sort_index()
        return result


