import pyximport
import warnings

warnings.filterwarnings('ignore', '.*deprecated NumPy.*')
pyximport.install(
    # setup_args={'include_dirs': np.get_include(), },
    reload_support=True,
)

# include_dirs = [
#     np.get_include(),
#     # os.path.dirname(__file__),
# ]
#
# extension = Extension(
#     name='cfuncs',
#     sources=['cfuncs.pyx'],
#     include_dirs=include_dirs,
#     define_macros=[('NPY_NO_DEPRECATED_API', 'NPY_1_7_API_VERSION')],
# )
#
# # setup(
# #     ext_modules=cythonize(
# #         extension,
# #         compiler_directives={'language_level': "3"},
# #     ),
# #     name='cfuncs',
# # )
#
from pluscodes.util.cfuncs import (
    encode_string,
    encode_digits,
    decode_digits,
)
