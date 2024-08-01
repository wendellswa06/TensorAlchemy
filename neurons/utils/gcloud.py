import json
from typing import Dict

from loguru import logger
from google.cloud import storage

from neurons.constants import (
    IA_BUCKET_NAME,
    IA_TEST_BUCKET_NAME,
)
from neurons.validator.config import get_config


def get_bucket_name() -> str:
    if get_config().netuid == "test":
        return IA_TEST_BUCKET_NAME

    return IA_BUCKET_NAME


async def get_storage_client() -> storage.Client:
    """
    Create and return an anonymous storage client.

    Returns:
        storage.Client: An anonymous storage client.
    """
    return storage.Client.create_anonymous_client()


def retrieve_public_file(
    source_name: str,
    bucket_name: str = get_bucket_name(),
    client: storage.Client = get_storage_client(),
) -> Dict:
    downloaded: Dict = None

    try:
        bucket = client.bucket(bucket_name)
        blob = bucket.blob(source_name)
        try:
            downloaded = json.loads(blob.download_as_text())
            logger.info(
                #
                f"Successfully downloaded {source_name} "
                + f"from {bucket_name}"
            )
        except Exception as e:
            logger.info(
                #
                f"Failed to download {source_name} from "
                + f"{bucket_name}: {e}"
            )

    except Exception as e:
        logger.info(f"An error occurred downloading from Google Cloud: {e}")

    return downloaded
