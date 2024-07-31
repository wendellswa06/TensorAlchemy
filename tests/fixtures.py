import pytest
import torch
import random
from PIL import Image, ImageDraw

from loguru import logger
import bittensor as bt

from neurons.protocol import ImageGeneration, ModelType
from neurons.utils.image import image_tensor_to_base64, image_to_tensor


class MockMetagraph:
    def __init__(self, n: int = 10) -> None:
        self.n = n
        self.hotkeys = [f"hotkey_{i}" for i in range(self.n)]
        self.coldkeys = [f"coldkey_{i}" for i in range(self.n)]


def mock_get_metagraph(n: int = 10):
    return MockMetagraph(n=n)


def generate_synapse(
    hotkey: str,
    image_content: torch.Tensor,
    prompt: str = "lion sitting in jungle",
    **kwargs,
) -> bt.Synapse:
    synapse = ImageGeneration(
        seed=-1,
        width=64,
        height=64,
        prompt=prompt,
        generation_type="TEXT_TO_IMAGE",
        model_type=ModelType.ALCHEMY.value,
        images=[image_tensor_to_base64(image_content)],
        **kwargs,
    )
    synapse.axon = bt.TerminalInfo(hotkey=hotkey)
    return synapse


def create_complex_image(size=(64, 64), num_blobs=3):
    image = Image.new(
        "RGB",
        size,
        color=(
            random.randint(0, 255),
            random.randint(0, 255),
            random.randint(0, 255),
        ),
    )
    draw = ImageDraw.Draw(image)

    for _ in range(num_blobs):
        blob_color = (
            random.randint(0, 255),
            random.randint(0, 255),
            random.randint(0, 255),
        )
        x = random.randint(0, size[0])
        y = random.randint(0, size[1])
        radius = random.randint(5, 20)
        draw.ellipse(
            [x - radius, y - radius, x + radius, y + radius], fill=blob_color
        )

    return image


TEST_IMAGES = {
    "BLACK": torch.zeros(
        [3, 64, 64],
        dtype=torch.float,
    ),
    "COMPLEX_A": image_to_tensor(create_complex_image()),
    "COMPLEX_B": image_to_tensor(create_complex_image()),
    "COMPLEX_C": image_to_tensor(create_complex_image()),
    "COMPLEX_D": image_to_tensor(create_complex_image()),
    "COMPLEX_E": image_to_tensor(create_complex_image()),
    "COMPLEX_F": image_to_tensor(create_complex_image()),
    "COMPLEX_G": image_to_tensor(create_complex_image()),
    # Fixed images which can be used for testing
    "ELEPHANT_BASKET": image_to_tensor(
        Image.open(r"tests/images/elephant_basket.png")
    ),
    "SPARROW_FISH": image_to_tensor(
        Image.open(r"tests/images/sparrow_fish.png")
    ),
    "EAGLE_FISH": image_to_tensor(
        Image.open(r"tests/images/eagle_fish.png"),
    ),
    "EAGLE_AMULET": image_to_tensor(
        Image.open(r"tests/images/eagle_amulet.png")
    ),
    "EAGLE_UMBRELLA": image_to_tensor(
        Image.open(r"tests/images/eagle_umbrella.png")
    ),
    "REAL_IMAGE": image_to_tensor(
        Image.open(r"tests/images/img.jpg"),
    ),
    "REAL_IMAGE_LOW_INFERENCE": image_to_tensor(
        Image.open(r"tests/images/img_low_inference_steps.jpg"),
    ),
}
