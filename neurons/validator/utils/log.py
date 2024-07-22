import json
import logging
import sys
from logging.handlers import QueueHandler, QueueListener
from multiprocessing import Queue

from loguru import logger

from neurons import constants
import bittensor as bt

import logging_loki


def get_subtensor_network_from_netuid(netuid: int) -> str:
    return {25: "testnet", 26: "finney"}.get(netuid, "")


def configure_loki_logger():
    from neurons.validator.config import get_config, validator_run_id

    """Configure sending logs to loki server"""

    if constants.IS_TEST:
        # Don't use loki for test runs
        return

    class LogHandler(logging_loki.LokiHandler):
        def handleError(self, record):
            self.emitter.close()
            # When Loki endpoint giving error for some reason,
            # parent .handleError starts spamming error trace for each failure
            # so we are disabling this default behaviour
            # super().handleError(record)

    class CustomLokiLoggingHandler(QueueHandler):
        def __init__(self, queue: Queue, **kwargs):
            super().__init__(queue)
            self.handler = LogHandler(**kwargs)  # noqa: WPS110
            self.listener = QueueListener(self.queue, self.handler)
            self.listener.start()

    class JSONFormatter(logging.Formatter):
        def format(self, record):
            from neurons.validator.utils.version import (
                get_validator_version,
                get_validator_spec_version,
            )

            try:
                # Extract real message noisy msg line emitted by bittensor
                # might exist better solution here
                msg = "".join(record.getMessage().split(" - ")[1:])
            except Exception:
                msg = record.getMessage()

            try:
                netuid = get_config().netuid
            except:
                netuid = ""

            try:
                hotkey = bt.wallet(config=get_config()).hotkey.ss58_address
            except Exception:
                hotkey = ""

            log_record = {
                "level": record.levelname.lower(),
                "module": record.module,
                "func_name": record.funcName,
                "thread": record.threadName,
                "run_id": validator_run_id.get(),
                "netuid": netuid,
                "subnet": get_subtensor_network_from_netuid(netuid),
                "hotkey": hotkey,
                "message": msg,
                "filename": record.filename,
                "lineno": record.lineno,
                "time": self.formatTime(record, self.datefmt),
                "version": get_validator_version(),
                "spec_version": get_validator_spec_version(),
            }
            return json.dumps(log_record)

    # Use LokiQueueHandler to upload logs in background
    loki_handler = CustomLokiLoggingHandler(
        Queue(-1),
        url="https://loki.tensoralchemy.ai/loki/api/v1/push",
        tags={"application": "tensoralchemy-validator"},
        auth=("tensoralchemy-loki", "tPaaDGH0lG"),
        version="1",
    )

    # Send logs to loki as JSON
    loki_handler.setFormatter(JSONFormatter())

    logger.add(loki_handler)


def patch_bt_logging():
    bt.logging.info = logger.info
    bt.logging.warning = logger.warning
    bt.logging.error = logger.error
    bt.logging.debug = logger.debug
    bt.logging.trace = logger.trace


def configure_logging():
    logger.remove()
    logger.add(
        sys.stdout,
        colorize=True,
        format="{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {name}:{function}:{line} - {message}",
    )
    configure_loki_logger()
    patch_bt_logging()
