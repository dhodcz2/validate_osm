# Collaboration of Validating entries across the globe
# The problem is: OSM entries are atomic, while the buildings are aggregates of them. How can we say something is
#   invalid, if the 'invalidity' was caused by some other piece's height, or lack thereof?
#   small piece has 3 height, other has NA,
#   only option is to make no statement if any NA; however this will disqualify many large, interesting buildings that
#   might be centers of attention because of their architectural feats
# GET https://validateosm.org/way/1234 ->
{
    'values': {'height': {
        'ChicagoBuildingFootprints': 3,
        'valid': 1
    },
        'floors': {
            'ChicagoBuildingFootprints': 1,
            'valid': 0,
        }
    },
    'authorities': {
        'ChicagoBuildingFootprints': 'dhodcz2@uic.edu',
    }

}

# authenticate with @edu
# POST https://validateosm.org/way/1234?height=4,
{
    'values': {'height': {
        'ChicagoBuildingFootprints': 3,
        'OtherSource': 3,
        'valid': 1
    },
        'floors': {
            'ChicagoBuildingFootprints': 1,
            'OtherSource': 2,
            'valid': 0,
        }
    },
    'authorities': {
        'ChicagoBuildingFootprints': 'dhodcz2@uic.edu',
        'OtherSource': 'JaneDoe@mit.edu'
    }
}

# TODO: This is from OpenCityModel; can instead use UBID?
# TODO: UBID is the standard for uniquely identifying building footprints on the Earth.
#   UBID for a building footprint has five components:
#   1   Open Location Code for the geometric center of mass (centroid) of the building footprint.
#   2   The distance to tne northern extent of the bounding box for the building foot in Open Location Code units
#   3   The distance to the eastern extent ...
#   4   The distance to the southern extent ...
#   5   The distance to the western extent ...

{
    'ubid': ...,
    'values': ...,
    'authorities': ...
    'contents': {
        'way/1234',
        'relation/1234',
    }
}

# Continue to flesh out Compare
