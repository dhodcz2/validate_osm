from typing import Optional

import matplotlib.axes._subplots
import matplotlib.pyplot as plt
from sources.nearest_neighbor import nearest_neighbor
import matplotlib.figure
import pandas as pd

plt.style.use('dark_background')

# from matches import true_matches, match_two_addresses

import matplotlib.pyplot as plt
import geopandas as gpd
from geopandas import GeoDataFrame

# def plot_discrepancies(matches: GeoDataFrame, validating: str, source: str):
#     matches = matches[
#         (pd.notna(matches[validating])) &
#         (pd.notna(matches[validating + '_hay']))
#         ]
#
#     dif = f'{validating} difference'
#     dif = f'{validating} dif'
#     matches[dif] = np.abs(matches[validating].sub(matches[validating + '_hay']).fillna(0))
#     discrepancies = matches[matches[dif] > 0]
#
#     fig, ax = plt.subplots()
#     fig
#     ax.axes.xaxis.set_visible(True)
#     ax.axes.yaxis.set_visible(True)
#
#     divider = make_axes_locatable(ax)
#     cax = divider.append_axes('bottom', size='1%', pad=0.1)
#     matches[~matches.index.isin(discrepancies.index)].exterior.plot(ax=ax, color='black')
#
#     discrepancies[dif] = discrepancies[dif].astype(int)
#     discrepancies.plot(
#         ax=ax, cmap='Reds', column=dif, edgecolor='black', legend=True, cax=cax,
#         # ax=ax, cmap='Reds', column=validating, markersize=1000, edgecolors='red', legend=True, cax=cax, linewidth=3,
#         legend_kwds={'label': f'Absolute value of difference between osm entry {validating} and {validating} according'
#                               f' to {source}',
#                      'orientation': 'horizontal'}
#     )
#     return fig, ax


from validate_osm.util.matches import match_two_addresses


def plot_highlight_mismatches(mismatches: GeoDataFrame, all_matches: GeoDataFrame):
    all_matches = all_matches[~all_matches.index.isin(mismatches.index)]
    fig, ax = plt.subplots(1, 1)
    all_matches.exterior.plot(ax=ax, color='black', figsize=[300, 150])
    mismatches.exterior.plot(ax=ax, color='red', figsize=[300, 150])
    return fig, ax


def plot_mismatch(mismatch: GeoDataFrame, haystack: GeoDataFrame, identifiers: list[str] = []):
    mismatch.geometry = mismatch.geometry.to_crs(3857)
    mismatch.geometry_hay = mismatch.geometry_hay.to_crs(3857)
    mismatch_gs = mismatch.iloc[0]
    haystack.geometry = haystack.geometry.to_crs(3857)
    buff = gpd.GeoDataFrame({
        'geometry': mismatch.geometry.buffer(100)
    }, crs=3857)
    surroundings = gpd.sjoin(haystack, buff, op='intersects', how='inner')

    def find_true_match(candidates: GeoDataFrame):
        for ident in identifiers:
            if ident not in candidates or ident not in mismatch_gs:
                continue
            true_match = candidates[candidates[ident] == mismatch_gs['bldg_id']]
            if len(true_match):
                return true_match
            else:
                candidates = candidates[~(candidates[ident] != mismatch_gs[ident])]
                # candidates = candidates[~(candidates[ident] != mismatch[ident])]
        true_match = candidates[
            candidates['address'].appliers(
                match_two_addresses, mismatch_gs['address']
            )
        ]
        return true_match

    true_match = find_true_match(surroundings)
    surroundings = surroundings[~surroundings.index.isin(true_match.index)]

    fig, ax = plt.subplots()
    # fig: plt.figure.F, ax = plt.subplots()

    surroundings.geometry.exterior.plot(ax=ax, color='black')
    mismatch.geometry.exterior.plot(ax=ax, color='blue')
    mismatch.geometry.centroid.plot(ax=ax, color='blue')
    mismatch.geometry_hay.exterior.plot(ax=ax, color='red')
    mismatch.geometry_hay.centroid.plot(ax=ax, color='red')

    string = f"OSM ID {mismatch_gs['osm_id']} mismatch;\n"
    for ident in identifiers:
        string += f'{ident}: {mismatch_gs[ident]}\n' if not pd.isna(mismatch_gs[ident]) else ''
    string += f"Address {mismatch_gs['address']}\n" if not pd.isna(mismatch_gs['address']) else ""
    fig.suptitle(string, color='blue', y=1.05)

    fig: matplotlib.figure.Figure
    ax: matplotlib.axes._subplots.AxesSubplot
    return fig, ax


def plot_mismatch(mismatch: GeoDataFrame, needles: GeoDataFrame, haystack: GeoDataFrame, identifiers: list[str] = []):
    mismatch.geometry = mismatch.geometry.to_crs(3857)
    mismatch.geometry_hay = mismatch.geometry_hay.to_crs(3857)
    haystack.geometry = haystack.geometry.to_crs(3857)
    needles.geometry = needles.geometry.to_crs(3857)

    fig, ax = plt.subplots(1, 2)
    ax_needle = ax[0]
    ax_hay = ax[1]
    ax_hay.axes.xaxis.set_visible(False)
    ax_hay.axes.yaxis.set_visible(False)
    ax_needle.axes.xaxis.set_visible(False)
    ax_needle.axes.yaxis.set_visible(False)

    buff = gpd.GeoDataFrame({
        'geometry': mismatch.geometry.buffer(50)
    }, crs=3857)

    needle_surroundings = gpd.sjoin(needles, buff)
    haystack_surroundings = gpd.sjoin(haystack, buff)

    haystack_surroundings.geometry.exterior.plot(ax=ax_hay, color='black')
    needle_surroundings.geometry.exterior.plot(ax=ax_needle, color='black')

    mismatch.geometry.exterior.plot(ax=ax_needle, color='blue')
    mismatch.geometry_hay.exterior.plot(ax=ax_hay, color='red')

    mismatch.geometry.centroid.plot(ax=ax_hay, color='blue')
    mismatch.geometry.centroid.plot(ax=ax_needle, color='blue')

    mismatch.geometry_hay.centroid.plot(ax=ax_hay, color='red')
    mismatch.geometry_hay.centroid.plot(ax=ax_needle, color='red')

    mismatch_gs = mismatch.iloc[0]

    def find_true_match(candidates: GeoDataFrame):
        true_match = gpd.GeoDataFrame()
        for id in identifiers:
            if len(true_match := candidates[candidates[id] == mismatch_gs[id]]):
                return true_match
            candidates = candidates[~(candidates[id] != mismatch_gs[id])]
        if pd.notna(mismatch_gs['address']):
            candidates = candidates[~(
                pd.isna(candidates['address'])
            )]
            true_match = candidates[candidates.appliers(
                match_two_addresses, haystalk=mismatch_gs['address'], axis=1
            )]
        return true_match

    true_match = find_true_match(haystack)
    try:
        true_match.geometry.exterior.plot(ax=ax_hay, color='green')
        true_match.geometry.centroid.plot(ax=ax_hay, color='green')
        true_match.geometry.centroid.plot(ax=ax_needle, color='green')
    except AttributeError:
        pass

    string = f"OSM ID {mismatch_gs['osm_id']} mismatch;\n"
    for ident in identifiers:
        string += f'{ident}: {mismatch_gs[ident]}\n' if pd.notna(mismatch_gs[ident]) else ''
    string += f"Address {mismatch_gs['address']}\n" if pd.notna(mismatch_gs['address']) else ""
    fig.suptitle(string, color='blue', y=1.30)

    string = ""
    for id in identifiers:
        id_hay = id + '_hay'
        string += f'{id}: {mismatch_gs[id_hay]}\n' if pd.notna(mismatch_gs[id_hay]) else ''
    string += f"Address {mismatch_gs['address_hay']}\n" if pd.notna(mismatch_gs['address_hay']) else ""
    string += f"Dist {mismatch_gs['distance']:.3f}\n"
    ax_needle.set_title(string, color='red')

    if len(true_match):
        # matched_hay = mismatch[(col for col in mismatch.columns if (str.endswith(col, '_hay')))]
        # matched_hay.columns = matched_hay.columns.str.replace(r'_hay', '')
        # matched_hay = GeoDataFrame(matched_hay, crs=3857)
        # matched_hay_gs = matched_hay.iloc[0]
        # compare = GeoDataFrame()
        # compare = compare.append(matched_hay)
        # compare = compare.append(true_match)

        distance = nearest_neighbor(mismatch, true_match, identifiers).iloc[0].distance
        # dist_true = closest.iloc[0].distance
        # dist_true = nearest_neighbor(mismatch, compare).iloc[0].distance

        true_match_gs = true_match.iloc[0]
        string = ""
        for ident in identifiers:
            string += f'{ident}: {true_match_gs[ident]}\n' if pd.notna(true_match_gs[ident]) else ''
        string += f"Address {true_match_gs['address']}\n" if pd.notna(true_match_gs['address']) else ""
        string += f"""Dist {distance:.3f}"""
        ax_hay.set_title(string, color='green')

    return fig, ax


def plot_discrepancies(gdf: gpd.GeoDataFrame, col: str):
    fig, ax = plt.subplots()
    ax.tick_params(labelsize=100)
    # ax.axes.xaxis.set_visible(True)
    # ax.axes.yaxis.set_visible(True)
    # divider = make_axes_locatable(ax)
    # cax = divider.append_axes('bottom', size='1%', pad=0.1)

    if len(gdf):
        gdf.plot(ax=ax, cmap='rainbow', column=col, legend=True, )
    fig.axes[1].tick_params(labelsize=100)

    if len(gdf):
        gdf.exterior.plot(ax=ax, color='black')

    return fig, ax


# def plot_mismatch(mismatch: GeoDataFrame, haystack: GeoDataFrame, identifiers: list[str] = []):
#     # plt.autoscale(True)
#
#     buff = gpd.GeoDataFrame({
#         'geometry': (mismatch.geometry.buffer(5),)
#     }, crs=3857)
#     haystack = haystack.to_crs(3857)
#     # buff = GeoDataFrame(geometry=mismatch.geometry.buffer(5))
#     surroundings = gpd.sjoin(haystack, buff, op='intersects', how='inner')
#
#     def find_true_match(candidates: GeoDataFrame):
#         for ident in identifiers:
#             if ident not in candidates or ident not in mismatch:
#                 continue
#             true_match = candidates[candidates[ident] == mismatch[ident]]
#             if len(true_match):
#                 return true_match
#             else:
#                 candidates = candidates[~(candidates[ident] != mismatch[ident])]
#         true_match = candidates[
#             candidates['address'].apply(
#                 match_two_addresses, mismatch['address']
#             )
#         ]
#         return true_match
#
#     true_match = find_true_match(surroundings)
#     surroundings = surroundings[~surroundings.index.isin(true_match.index)]
#
#     # for ident in identifies:
#     #      if
#     #      true_matches = candidates[candidates[ident] == mismatch[ident]]
#     #      if len(true_matches):
#     #          return true_matches
#     #
#     #      candidates = candidates[pd.isna(candidates[ident]) | ]
#
#     # candidates: GeoDataFrame = surroundings[
#     #     pd.isna(surroundings[iden])
#     # ]
#
#     base = surroundings.geometry.exterior.plot(color='black')
#     # mismatch.geometry.exterior.plot(ax=zcbase, color='blue')
#     # mismatch.loc[[0], 'geometry'].extero
#     base.plot(*mismatch.geometry.exterior.xy, color='blue')
#     base.plot(mismatch.geometry.centroid.xy, color='blue')
#     base.plot(*mismatch.geometry_hay.exterior.xy, color='red')
#     base.plot(mismatch.geometry_hay.centroid.xy, color='red')
#
#     # mismatch.point.plot(ax=base, color='blue')
#     # mismatch.geometry_hay.exterior.plot(ax=base, color='red')
#     # mismatch.point_hay.exterior.plot(ax=base, color='red')
#     if len(true_match):
#
#         true_match.geometry.exterior.plot(ax=base, color='green')
#         true_match.point.plot(ax=base, color='green')
#
#     string = f"osm ID {mismatch['osm_id']} mismatch;\n"
#     for ident in identifiers:
#         string += f'{ident}: {mismatch[ident]}\n' if not pd.isna(mismatch[ident]) else ''
#     string += f"Address {mismatch['address']}\n" if not pd.isna(mismatch['address']) else ""
#
#     return base
def plot_discrepancy(gdf: GeoDataFrame, match: GeoDataFrame, r_suffix: str = '_r'):
    fig, ax = plt.subplots(1, 2)
    ax_needle: matplotlib.axes.Axes = ax[0]
    ax_hay: matplotlib.axes.Axes = ax[1]
    ax_hay.axes.xaxis.set_visible(False)
    ax_hay.axes.yaxis.set_visible(False)
    ax_needle.axes.xaxis.set_visible(False)
    ax_needle.axes.yaxis.set_visible(False)

    geometry: gpd.GeoSeries = match['geometry']
    geometry_r: gpd.GeoSeries = match['geometry' + r_suffix]

    needle_surroundings: GeoDataFrame = gdf.loc[
        gdf['geometry'].intersects(
            geometry.iloc[0].buffer(.0005)
        )
    ]
    haystack_surroundings: GeoDataFrame = gdf.loc[
        gdf['geometry' + r_suffix].intersects(
            geometry_r.iloc[0].buffer(.0005)
        )
    ]

    needle_surroundings['geometry'].exterior.plot(ax=ax_needle, color='green')
    geometry.plot(ax=ax_needle, color='yellow')
    for point, height in zip(needle_surroundings['centroid'].to_crs(4326), needle_surroundings['height_m']):
        ax_needle.annotate(str(int(round(height, 0))), (float(point.x), float(point.y)))

    haystack_surroundings['geometry' + r_suffix].geometry.exterior.plot(color='green', ax=ax_hay)
    geometry_r.plot(ax=ax_hay, color='yellow')
    for point, height in zip(haystack_surroundings['centroid' + r_suffix].to_crs(4326),
                             haystack_surroundings['height_m' + r_suffix]):
        ax_hay.annotate(str(int(round(height, 0))), (float(point.x), float(point.y)))


def plot_discrepancy(match: gpd.GeoDataFrame, needles: gpd.GeoSeries, haystack: gpd.GeoSeries):
    fig, ax = plt.subplots(1, 2)
    ax: list[matplotlib.axes.Axes]
    ax_needle = ax[0]
    ax_hay = ax[1]
    ax_hay.axes.xaxis.set_visible(False)
    ax_hay.axes.yaxis.set_visible(False)
    ax_needle.axes.xaxis.set_visible(False)
    ax_needle.axes.yaxis.set_visible(False)

    needle_surroundings: GeoDataFrame = needles.loc[
        needles['geometry'].intersects(
            match['centroid'].buffer(100).to_crs(4326).iloc[0]
        )
    ]
    haystack_surroundings: GeoDataFrame = haystack.loc[
        haystack['geometry'].intersects(
            match['centroid_r'].buffer(100).to_crs(4326).iloc[0]
        )
    ]

    needle_surroundings['geometry'].exterior.plot(ax=ax_needle, color='green')
    match['geometry'].plot(ax=ax_needle, color='yellow')
    for point, height in zip(needle_surroundings['centroid'].to_crs(4326), needle_surroundings['height_m']):
        if pd.isna(height):
            ax_needle.annotate('NA', (float(point.x), float(point.y)))
        else:
            ax_needle.annotate(str(int(round(height, 0))), (float(point.x), float(point.y)))

    haystack_surroundings['geometry'].exterior.plot(ax=ax_hay, color='green')
    match['geometry_r'].plot(ax=ax_hay, color='yellow')
    for point, height in zip(haystack_surroundings['centroid'].to_crs(4326), haystack_surroundings['height_m']):
        if pd.isna(height):
            ax_needle.annotate('NA', (float(point.x), float(point.y),))
        else:
            ax_hay.annotate(str(int(round(height, 0))), (float(point.x), float(point.y)))


def plot_groupings(
        source: GeoDataFrame,
        recipient: GeoDataFrame,
        source_index_in_recip: str,
        bbox: Optional[tuple] = None,
        plot_hatches=(False, True),
        flip_ax=False
):
    plt.style.use('dark_background')
    fig, (ax_s, ax_r) = plt.subplots(1, 2)
    if flip_ax:
        ax_r, ax_s = ax_s, ax_r
        plot_hatches = (plot_hatches[1], plot_hatches[0])
    # plt.rcParams['figure.figsize'] = (100, 100)
    if bbox is not None:
        recipient = recipient.cx[bbox[0]:bbox[2], bbox[1]:bbox[3]]
    ungrouped_recipient: GeoDataFrame = recipient[recipient[source_index_in_recip].isna()]
    recipient: GeoDataFrame = recipient.loc[recipient.index.difference(ungrouped_recipient.index)]
    ungrouped_source: GeoDataFrame = source.loc[source.index.difference(recipient[source_index_in_recip])]
    source: GeoDataFrame = source.loc[source.index.difference(ungrouped_source.index)]
    if bbox is not None:
        ungrouped_source = ungrouped_source.cx[bbox[0]:bbox[2], bbox[1]:bbox[3]]

    import matplotlib.colors as mcolors
    colors = list(mcolors.TABLEAU_COLORS.values())
    hatches = '\ - | \\'.split()
    source['color'] = pd.Series((
            colors * (len(source) // len(colors)) + colors[:(len(source) % len(colors))]
    ), index=source.index)
    source['hatch'] = pd.Series((
            hatches * (len(source) // len(hatches)) + hatches[:(len(source) % len(hatches))]
    ), index=source.index)
    recipient['color'] = source.loc[recipient[source_index_in_recip], 'color'].values
    recipient['hatch'] = source.loc[recipient[source_index_in_recip], 'hatch'].values

    if len(ungrouped_recipient):
        ungrouped_recipient.exterior.plot(color='white', ax=ax_r)
    if len(ungrouped_source):
        ungrouped_source.exterior.plot(color='grey', hatch='*', ax=ax_s)
    for (color, hatch), index in recipient.groupby(['color', 'hatch']).groups.items():
        recipient.loc[index].exterior.plot(
            color=color,
            hatch=hatch if plot_hatches[1] else None,
            ax=ax_r
        )
    for (color, hatch), index in source.groupby(['color', 'hatch']).groups.items():
        source.loc[index].exterior.plot(
            color=color,
            hatch=hatch if plot_hatches[0] else None,
            ax=ax_s
        )


if __name__ == '__main__':
    needles_combined = gpd.read_feather(
        '/coalescer/cache/NewYorkHeightNeedles.feather')
    haystack_combined = gpd.read_feather(
        '/coalescer/cache/NewYork3DModel.feather')
    from comparison import percent_error

    gdf = needles_combined.merge(haystack_combined, on='bin', how='inner', suffixes=[None, '_r'])
    gdf = gdf[
        pd.notna(gdf['height_m']) & pd.notna(gdf['height_m_r'])
        ]
    # %%
    gdf['percent_error'] = gdf.appliers(percent_error, nonauthoritative_col='height_m', authoritative_col='height_m_r',
                                        axis=1)
    gdf['percent_error'] = gdf['percent_error'].round(4)
    gdf = gdf[:100]
    plot_discrepancies(gdf, 'percent_error')
