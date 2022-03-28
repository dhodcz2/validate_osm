class DescriptorDump:
    {
        'A': {
            'id': {
                'w123', 'r123'
            },
            'osm': 4,
            'cbf': 5
        },
        'B': {
            'id': {
                'w1234',
            },
            'osm': 6,

        }
    }
    """
    
    """
    """
    Professor Miranda, I'm having trouble determining how the percent_difference results
    should be structured. I want something like compare.to_json or compare.to_csv, but neither
    csv nor json seem appropriate. We want to prioritize high difference buildings, so I think that the output
    should be sorted, which rules out JSON. 
    
    osm,cbf
    4,44
    30,3
    1,2,
    1,1
    
    
    """
    identities = {
        'A':
    }
    def __get__(self, instance, owner):
        ...

    def __call__(self, *args, **kwargs):
        ...


