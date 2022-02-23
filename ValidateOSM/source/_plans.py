'''
Currently the pipeline is something like:
static | data | group | aggregate
We want to upgrade this progress so that datasets are conscious of UBID from one another

static | data | group | aggregate | identify


static | data | group | aggregate |


static | data | group | aggregate | exclude | identify -> batch


static  ~
data        serialize dataframe
group       serialize a series that includes integers that indicate groups
aggregate   serialize dataframe
exclude     serialize a dataframe of those that are to be excluded from aggregate
identify    serialize a series
...
TODO: Is there benefit in serializing the 'groups'?
    perhaps for uniformity of processes
i'''

'''
static
data        dataframe.to_Feather        -> GeoDataFrame
group       sqlite                      -> series[int]
aggregate   dataframe.to_feather        -> GeoDataFrame
identity    sqlite                      -> series[obj]
exclude     sqlite                      -> set[str]
batch       dataframe.to_feather        -> GeoDataFrame
'''

'''

static
data        gdf     to_file
group       gdf
aggregate   gdf
identity     gdf
exclude     gdf
batch       gdf     to_file
"""
It seems like it would be worthwhile to optimize disk usage. i
Source.aggregate    ->  data[group.isna] U aggregate
batch   ->              Source.aggregate[identity] = identity; Source.loc[~exclude]
"""


{
    'height': {
        'EVL_OSM': 1,
        'ChicagoBuildingFootprints': 2,
        'consensus': -1
    },
    'osm': {
        'EVL_OSM': {
            'way/123',
            'way/1234',
            'relation/123'
        },
        'MIT_OSM': {
            'way/123',
            'relation/123',
        },
    },
}

# TODO: Source.footprint.data automatically overlaps the two BBoxes and then only loads within that BBox

