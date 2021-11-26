from collections import Generator
from datetime import datetime
import geopandas as gpd
import pandas as pd
import numpy as np

"""
1. Histogram of ID percent_error (Intersect)
2. Histogram of distance percent_error (Intersect)
3. Histogram of ID corrected with distance (Intersect)
4. Histogram of ID corrected with distance (Union)
5. Histogram of ID, not corrected with distance (Union)
"""


def percent_difference(gdf: gpd.GeoDataFrame, col: str = 'height_m'):
    r_col = col + '_r'
    return (
            np.abs(gdf[col] - gdf[r_col]) /
            ((gdf[col] + gdf[r_col]) / 2)
    )


def percent_error_normalized(gdf: gpd.GeoDataFrame, col='height_m') -> Generator[float]:
    for height_left, height_right in zip(gdf[col], gdf[col + '_r']):
        yield (
                (height_right - height_left) / height_right
        ) if height_right > height_left else (
                (height_left - height_right) / height_left
        )


def preprocess_needles(needles: gpd.GeoDataFrame):
    needles = needles[~(needles['start_date'] > datetime(2014, 1, 1, ))]
    needles = needles[pd.notna(needles['height_m'])]
    return needles


def exclusions(gdf: gpd.GeoDataFrame):
    gdf = gdf[gdf['geometry'].intersects(gdf['geometry_r'])]
    gdf = gdf[pd.notna(gdf['height_m'])]
    gdf = gdf[pd.notna(gdf['height_m_r'])]
    return gdf


def merge_on_identifier(left: gpd.GeoDataFrame, right: gpd.GeoDataFrame, on: str, suffixes=[None, '_r'], **kwargs):
    left = left[left[on].notna()]
    left = left.reset_index()
    right = right[right[on].notna()]
    gdf = left.merge(right, on=on, how='inner', suffixes=[None, '_r'])
    gdf = gdf.set_index('index', True)
    # TODO: Once again we fucked up because it should be list not pd.Series :(
    gdf = exclusions(gdf)
    gdf = gdf.assign(percent_error=np.fromiter(percent_error_normalized(gdf), dtype=float))
    return gdf


def merge_on_distance(left: gpd.GeoDataFrame, right: gpd.GeoDataFrame):
    left = left.set_geometry('centroid')
    right = right.set_geometry('centroid')
    gdf = left.sjoin_nearest(right, how='left', lsuffix='', rsuffix='r', distance_col='dist', )
    gdf.columns = gdf.columns.str.rstrip('_')
    gdf = gdf.set_geometry('geometry')
    gdf = gdf.merge(right['centroid'], suffixes=[None, '_r'], left_on='index_r', right_index=True, how='left')
    gdf = exclusions(gdf)
    gdf = gdf.assign(percent_error=np.fromiter(percent_error_normalized(gdf), dtype=float))
    return gdf


#
# def merge_on_dist_id_comparison(left: gpd.GeoDataFrame, right: gpd.GeoDataFrame, on: str, suffixes=[None, '_r'],
#                                 **kwargs):
#     merge_id = merge_on_identifier(left, right, on, suffixes, **kwargs)
#     merge_dist = merge_on_distance(left, right, suffixes, **kwargs)
#     df = (
#         merge_id[[on, 'percent_error']]
#             .assign(index=merge_id.index)
#             .merge(
#             merge_dist[[on, 'percent_error']]
#                 .assign(index=merge_dist.index),
#             how='inner', on=on, suffixes=['_id', '_dist']
#         )
#     )
#     include_from_id_and_exclude_from_dist: list[tuple[int, int]] = [
#         (id_index, dist_index)
#         for id_error, id_index, dist_error, dist_index in zip(
#             df['percent_error_id'], df['index_id'], df['percent_error_dist'], df['index_dist']
#         ) if id_error < dist_error
#     ]
#     merge_id = merge_id.loc[(id for id, _ in include_from_id_and_exclude_from_dist)]
#     merge_dist = merge_dist.loc[
#         merge_dist.index.difference((id[0] for id in include_from_id_and_exclude_from_dist))
#     ]
#     return merge_id.append(merge_dist)
#
#
def merge_on_dist_id_comparison(left: gpd.GeoDataFrame, right: gpd.GeoDataFrame, on: str):
    merge_id = merge_on_identifier(left, right, on)
    merge_dist = merge_on_distance(left, right)
    comparison = merge_id[['percent_error']].join(
        merge_dist[['percent_error']], how='inner', lsuffix='_id', rsuffix='_dist'
    )
    # We prefer the ID if errors are identical
    indices = [
        i
        for i, error_id, error_dist in zip(
            comparison.index, comparison['percent_error_id'], comparison['percent_error_dist']
        ) if error_dist < error_id
    ]
    result: gpd.GeoDataFrame = merge_id.drop(indices)
    result = result.append(merge_dist.loc[merge_dist.index.difference(result.index)])
    result = result.sort_index()
    return result
    # result: gpd.GeoDataFrame = merge_id.loc[indices]
    # merge_dist = merge_dist.drop(indices)
    # result = result.append(merge_dist)
    # result = result.sort_index()
    # return result

    # merge_id = merge_on_identifier(left, right, on)
    # merge_dist = merge_on_distance(left, right)
    # left: pd.DataFrame = merge_id[['percent_error']]
    # left = left.assign(backref=left.index.get_level_values(0)).reset_index(drop=True)
    # right: pd.Series = merge_dist[['percent_error']]
    # right = right.assign(backref=right.index.get_level_values(0)).reset_index(drop=True)
    # df = left.merge(right, how='inner', suffixes=['_id', '_dist'], on='backref')
    # include_from_id_and_exclude_from_dist: list[int] = [
    #     i
    #     for i, error_id, error_dist in zip(
    #         df['backref'], df['percent_error_id'], df['percent_error_dist']
    #     ) if error_id < error_dist
    # ]
    # merge_id = merge_id.loc[include_from_id_and_exclude_from_dist]
    # merge_dist = merge_dist.drop(include_from_id_and_exclude_from_dist)
    # return merge_id.append(merge_dist)
    #

#     merge_id = merge_on_identifier(left, right, on, suffixes, **kwargs)
#     merge_dist = merge_on_distance(left, right, suffixes, **kwargs)
#     # merge_dist = merge_on_distance(left.loc[merge_id.index], right, suffixes, **kwargs)
#     df = (
#         merge_id[[on, 'percent_error']]
#             .assign(index=merge_id.index)
#             .merge(
#             merge_dist[[on, 'percent_error']]
#                 .assign(index=merge_dist.index),
#             how='inner', on=on, suffixes=['_id', '_dist']
#         )
#     )
#     include_from_id_and_exclude_from_dist: list[tuple[int, int]] = [
#         (id_index, dist_index)
#         for id_error, id_index, dist_error, dist_index in zip(
#             df['percent_error_id'], df['index_id'], df['percent_error_dist'], df['index_dist']
#         ) if id_error < dist_error
#     ]
#     merge_id = merge_id.loc[(id for id, _ in include_from_id_and_exclude_from_dist)]
#     merge_dist = merge_dist.loc[
#         set(merge_dist.index).difference((id for _, id in include_from_id_and_exclude_from_dist))
#     ]
#     return merge_id.append(merge_dist)
# # %%
