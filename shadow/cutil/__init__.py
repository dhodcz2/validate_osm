import warnings

import numpy as np
import pyximport

warnings.filterwarnings('ignore', '.*deprecated NumPy.*')
pyximport.install(
    setup_args={'include_dirs': np.get_include(), },
    reload_support=True,
)
from shadow.cutil.cutil import (
    load_image,
    num2deg,
    deg2num,
    degs2nums,
    nums2degs,
)
