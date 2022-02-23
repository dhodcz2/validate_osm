from argparse import ArgumentParser


class _Namespace:
    debug: bool


_argparser = ArgumentParser()
_argparser.add_argument('-f', )  # Workaround for ipython
_argparser.add_argument('--debug', dest='debug', action='store_true')
global_args = _argparser.parse_args(namespace=_Namespace)
