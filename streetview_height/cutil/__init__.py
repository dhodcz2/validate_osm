import numpy as np

from typing import Callable
import pyximport

pyximport.install(
    setup_args={'include_dirs': np.get_include(), },
    reload_support=True,
)
from streetview_height.cutil.util import (
    cdisplacement,
    deg2num,
    num2deg,
    # ppdeg2num,
    # ppnum2deg
    _deg2num,
    _num2deg,
)
