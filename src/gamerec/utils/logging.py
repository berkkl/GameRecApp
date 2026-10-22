import logging


def setup_logging():
    logging.basicConfig(level=logging.INFO)
    return logging.getLogger("gamerec")
