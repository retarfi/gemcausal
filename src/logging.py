import logging

logger: logging.Logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
ch: logging.StreamHandler = logging.StreamHandler()
ch.setLevel(logging.INFO)
formatter: logging.Formatter = logging.Formatter(
    fmt="%(asctime)s [%(levelname)s] %(message)s", datefmt="%Y/%m/%d %H:%M:%S"
)
ch.setFormatter(formatter)
logger.addHandler(ch)
