import logging
from logging import NullHandler


def set_console_logger(level=logging.DEBUG, format_string="%(asctime)s %(name)s [%(levelname)s]: %(message)s"):
    log.setLevel(level)
    sh = logging.StreamHandler()
    sh.setLevel(level)
    formatter = logging.Formatter(format_string)
    sh.setFormatter(formatter)
    log.addHandler(sh)


log = logging.getLogger(__name__)
if not log.handlers:
    log.addHandler(NullHandler())
