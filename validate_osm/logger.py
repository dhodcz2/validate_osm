import time
from contextlib import contextmanager
import logging

import python_log_indenter

logger = logging.getLogger(__name__.partition('.')[0])
for handler in logger.handlers:
    if isinstance(handler, logging.StreamHandler):
        break
else:
    logger.setLevel(logging.DEBUG)
    handler = logging.StreamHandler()
    formatter = logging.Formatter(f'%(levelname)-10s %(message)s')
    handler.setFormatter(formatter)
    logger.addHandler(handler)

logger = python_log_indenter.IndentedLoggerAdapter(logger)


@contextmanager
def logged_subprocess(message: str, level=logging.INFO, timed=True):
    match level:
        case logging.DEBUG:
            logger.debug(message)
        case logging.INFO:
            logger.info(message)
        case logging.WARN:
            logger.warning(message)
        case logging.ERROR:
            logger.error(message)
        case logging.CRITICAL:
            logger.critical(message)

    logger.add()
    t = time.time()
    yield
    t = (time.time() - t) / 60
    message = f'{t:.1f} minutes'
    if timed:
        match level:
            case logging.DEBUG:
                logger.debug(message)
            case logging.INFO:
                logger.info(message)
            case logging.WARN:
                logger.warning(message)
            case logging.ERROR:
                logger.error(message)
            case logging.CRITICAL:
                logger.critical(message)
    logger.sub()
