from typing import Type

from pandas import IndexSlice as idx
import pandas as pd
from geopandas import GeoSeries, GeoDataFrame
import dataclasses

if False:
    from validate_osm import Compare


@dataclasses.dataclass
class StructIdentities:
    dict: dict


@dataclasses.dataclass
class StructDiscrepancies:
    dict: dict


@dataclasses.dataclass
class StructTickets:
    ...


class DescriptorOSM:
    def __get__(self, instance: 'Compare', owner: Type["Compare"]):
        self.compare = instance
        self.owner = owner
        return self

    @property
    def identities(self):
        data: GeoDataFrame = self.compare.data.xs('osm', level='name')
        ids = data.index.get_level_values('id')
        result = {
            identity: ids[ilocs]
            for identity, ilocs in data.groupby(self.compare.identity).indices.items()
        }
        return StructIdentities(result)

    @property
    def discrepancies(self):
        diff = self.compare.matrix.percent_difference(
            rows=self.compare.names.difference('osm'),
            columns='osm',
            value='height_m'
        )
        diff: pd.Series = diff.loc[diff['osm'] > .20, 'osm']
        diff.sort_values(ascending=False)

        identifiers = diff.index.get_level_values(self.compare.identity)
        osm: pd.DataFrame = self.compare.aggregate.loc[idx[identifiers, 'osm'], :].reset_index(level='name', drop=True)
        others = self.compare.aggregate.loc[diff.index, 'height_m'].reset_index(level='name', drop=True)

        # discrepancies = pd.DataFrame({'osm': osm, 'height_m': others, })
        discrepancies = pd.DataFrame({
            'osm': osm['height_m'],
            # 'floors': osm['floors'],
            'height_m': others,
            'cardinal': osm['cardinal'],
            'ref': osm['ref'],
            'address': osm['address']
        })
        return discrepancies

    @property
    def cardinals(self):
        ...

    @property
    def tickets(self):
        ...
