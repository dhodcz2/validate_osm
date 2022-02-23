ValidateOSM is a framework to validate data from the free, editable geographic database OpenStreetMap (OSM). This is
accomplished by retrieving other publicly available datasets that contain data relevant to an OSM tag and make
comparisons to draw conclusions on the likelihood that a particular OSM entry is erroneous. Ultimately, project seeks to
limit the number of OSM entries that must be manually investigated, and improve the reliability and validity of
investigations that wish to leverage the free, abundant data available on OSM.

# Table of Contents

1. [Installation](#installation)
2. [Methodology](#methodology)
    1. [Source](#meth-source)
    3. [Compare](#meth-compare)
3. [API](#api)
    1. [Source](#api-source)
    2. [Compare](#api-compare)

# Installation

`pip install -e git=https://github.com/dhodcz2/ValidateOSM#egg=validateosm`</br>

# Methodology

## Source

The extraction of a source may be simplified as a data pipeline in which the raw data is standardized into uniform data;
the data is then grouped into many-to-one relationships; finally the groups are aggregated into single-value entries
which have one-to-one relations across other datasets:</br>
`raw | data | group | aggregate`

## Compare

**TODO**

# API

## Source

### Source</br>

#### Source.raw(self)

#### ValidateOSM.source.data

#### ValidateOSM.source.group

#### ValidateOSM.source.aggregate

### SourceOSM</br>

#### ValidateOSM.source.enumerate

##

Once the Sources within a .py have been defined, the input at runtime controls how Source.aggregate is to be determined:
`SourceOSM.aggregate()`
`SourceOSM.aggregate(Source1)'
`SourceOSM.aggregate(Source2)`
Here SourceOSM is group independently, and the others are grouped dependent to the geometry of SourceOSM.

## Compare

**TODO**



