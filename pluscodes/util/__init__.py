import cython
# from pluscodes.util.funcs import *
import warnings
import numpy as np
import pyximport

if True:
    warnings.filterwarnings('ignore', '.*deprecated NumPy.*')
    pyximport.install(
        setup_args=dict(include_dirs=np.get_include()),
        reload_support=True,
    )

    from pluscodes.util.cfuncs import (
        get_codes,
        get_digits,
    )
if __name__ == '__main__':
    ...

