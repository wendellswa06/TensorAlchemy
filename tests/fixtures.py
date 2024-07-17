import torch
from PIL import Image

from neurons.utils.image import image_to_tensor

TEST_IMAGES = {
    "BLACK": torch.zeros(
        [3, 1024, 1024],
        dtype=torch.float,
    ),
    "SOLID_COLOR_FILLED": torch.full(
        [3, 1024, 1024],
        0.2196,
        dtype=torch.float,
    ),
    "REAL_IMAGE": image_to_tensor(Image.open(r"tests/images/img.jpg")),
    "REAL_IMAGE_LOW_INFERENCE": image_to_tensor(
        Image.open(r"tests/images/img_low_inference_steps.jpg")
    ),
}
