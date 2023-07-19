import logging

from pytest import LogCaptureFixture

from src.logging import logger


def test_logger(caplog: LogCaptureFixture) -> None:
    src_log: str = "src.logging"
    logger.info("Info")
    assert ("src.logging", logging.INFO, "Info") in caplog.record_tuples
    logger.warning("Warning")
    assert ("src.logging", logging.WARN, "Warning") in caplog.record_tuples
    logger.error("Error")
    assert ("src.logging", logging.ERROR, "Error") in caplog.record_tuples
