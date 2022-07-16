from validate_osm.source.aggregate import FactoryAggregate
from validate_osm.source.bbox import BBox
from validate_osm.source.data import (
    DescriptorData,
    DecoratorData
)
from validate_osm.source.overpass import (
    DynamicOverpassResource,
    DecoratorEnumerative,
)
from validate_osm.source.resource_ import (
    StructFile,
    StructFiles,
    DescriptorStaticRegions,
    DescriptorStaticNaive,
    Resource
)
from validate_osm.source.source import Source
from validate_osm.source.source_osm import SourceOSM