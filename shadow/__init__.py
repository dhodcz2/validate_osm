import cython
import warnings

import numpy as np
import pyximport
warnings.filterwarnings('ignore', '.*deprecated NumPy.*')
pyximport.install(
    setup_args={'include_dirs': np.get_include(), },
    reload_support=True,
)
from shadow.dask_elevation_map import run