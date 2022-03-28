from pandas import IndexSlice as idx
import pandas as pd
from geopandas import GeoSeries, GeoDataFrame
import dataclasses


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
    def __get__(self, instance, owner):
        self._instance = instance
        self._owner = owner
        return self

    @property
    def identities(self):
        from validate_osm.compare.compare import Compare
        compare: Compare = self._instance
        data: GeoDataFrame = compare.data.xs('osm', level='name')
        ids = data.index.get_level_values('id')
        result = {
            identity: ids[ilocs]
            for identity, ilocs in data.groupby(compare.identity).indices.items()
        }
        return StructIdentities(result)

    @property
    def discrepancies(self):
        from validate_osm.compare.compare import Compare
        compare: Compare = self._instance
        diff = compare.matrix.percent_difference(rows=compare.names.difference('osm'), columns='osm', value='height_m')
        diff: pd.Series = diff.loc[diff['osm'] > .20, 'osm']
        diff.sort_values(ascending=False)

        identifiers = diff.index.get_level_values(compare.identity)
        osm: pd.DataFrame = compare.aggregate.loc[idx[identifiers, 'osm'], :].reset_index(level='name', drop=True)
        others = compare.aggregate.loc[diff.index, 'height_m'].reset_index(level='name', drop=True)

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
