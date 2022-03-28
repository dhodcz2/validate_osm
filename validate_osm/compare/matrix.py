import dataclasses
import itertools
from typing import Type, Optional, Any, Generator, Iterator, Union

import pandas as pd
from pandas import DataFrame


class DescriptorExport:
    json = {
        'A': {
            'cbf': 1,
            'osm': 2,
        },
        'B': {
            'cbf': 1,
            'osm': 2,
        },
    }
    csv = [
        ['ubid', 'cbf', 'osm'],
        ['A', 1, 2],
        ['B', 1, 2],
    ]


class DescriptorMatrix:
    def __init__(self):
        ...

    def __get__(self, instance, owner):
        self._instance = instance
        self._owner = owner
        return self

    def __enter__(self):
        ...

    def __exit__(self, exc_type, exc_val, exc_tb):
        # TODO: Send results to output stream.
        ...

    # TODO: Percent difference prioritizes floor, as

    def percent_difference(self, rows, columns, values: Union[str, list[str]]):
        if isinstance(values, str):
            values = (values,)
        else:
            values = list(values)

        # TODO: This doesn't seem to be sorted.
        from validate_osm.compare.compare import Compare
        compare: Compare = self._instance
        if isinstance(columns, (str, int)):
            columns = [columns, ]
        else:
            columns = list(columns)
        columns.sort()
        columns: list

        columns: dict[str, pd.DataFrame] = {
            source_name: compare.xs(source_name)
            for source_name in columns
        }

        if isinstance(rows, (str, int)):
            rows = [rows, ]
        else:
            rows = list(rows)
        rows.sort()
        rows: list

        rows: dict[str, pd.DataFrame] = {
            source_name: compare.xs(source_name)
            for source_name in rows
        }

        def column(cname) -> pd.Series:
            cdf = columns[cname]

            def gen():
                for rname, rdf in rows.items():
                    results: Iterator[pd.Series] = (
                        abs(cdf[value] - rdf[value]) /
                        pd.DataFrame({'c': cdf[value], 'r': rdf[value]}).max(axis=1)
                        for value in values
                    )
                    result = next(results)
                    for r in results:
                        result.update(r.loc[result[result.isna()].index])
                    result.index = pd.MultiIndex.from_tuples(
                        zip(result.index, itertools.repeat(rname)), names=(result.index.name, 'name')
                    )
                    result.name = cname
                    result = result[result.notna()]
                    result = result.astype('float64')
                    yield result

            result: pd.Series = pd.concat(gen())
            return result

        result = pd.DataFrame({
            name: column(name)
            for name in columns.keys()
        })
        return result

        # TODO: Better to construct columnwise than rowwise
