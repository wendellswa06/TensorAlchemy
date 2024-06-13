import asyncio
import pathlib
import sys

import sentry_sdk
from loguru import logger

from neurons.update_checker import check_for_updates

REPO_URL = "TensorAlchemy/TensorAlchemy"

if __name__ == "__main__":
    # Add the base repository to the path so the validator can access it
    file_path = str(pathlib.Path(__file__).parent.parent.parent.resolve())
    if file_path not in sys.path:
        sys.path.append(file_path)
    current_folder = str(pathlib.Path(__file__).parent.resolve())

    try:
        check_for_updates(current_folder, REPO_URL)
    except Exception as error:
        logger.warning(
            "Failed to check remote for updates: " + str(error),
        )

    # Import StableValidator after fixing paths
    from neurons.constants import VALIDATOR_SENTRY_DSN

    from validator import StableValidator

    sentry_sdk.init(dsn=VALIDATOR_SENTRY_DSN)

    asyncio.run(StableValidator().run())
