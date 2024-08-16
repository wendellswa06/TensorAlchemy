import uuid
from typing import List

import bittensor as bt
from loguru import logger
from PIL.Image import Image as ImageType

from neurons.utils.image import image_to_base64
from neurons.protocol import ImageGeneration, ModelType

from neurons.config import clients

from scoring.types import ScoringResults
from scoring.pipeline import get_scoring_results


def generate_synapse(prompt: str, image_base64: str) -> bt.Synapse:
    hotkey: str = str(uuid.uuid4())
    try:
        synapse = ImageGeneration(
            seed=-1,
            width=1024,
            height=1024,
            prompt=prompt,
            images=[image_base64],
            generation_type="TEXT_TO_IMAGE",
            model_type=ModelType.CUSTOM.value,
        )
        synapse.axon = bt.TerminalInfo(hotkey=hotkey)
        logger.success("Successfully generated synapse")
        return synapse
    except Exception as e:
        logger.error(f"Error in generating synapse: {str(e)}")
        raise


async def score_images(synapses: List[bt.Synapse]) -> ScoringResults:
    logger.info(f"Scoring images for {len(synapses)} synapses")
    try:
        results = await get_scoring_results(
            ModelType.CUSTOM, synapses[0], synapses
        )
        logger.success("Successfully scored images.")
        return results
    except Exception as e:
        logger.error(f"Error in scoring images: {str(e)}")
        raise


async def score_imageset(
    prompt: str, images: List[ImageType]
) -> ScoringResults:
    synapses = []

    for image in images:
        synapses.append(generate_synapse(prompt, image_to_base64(image)))

    class MetagraphMock:
        "We will pass in a mocked metagraph"
        n = len(synapses)
        hotkeys = [s.axon.hotkey for s in synapses]

    clients.metagraph = MetagraphMock()

    return await score_images(synapses)
