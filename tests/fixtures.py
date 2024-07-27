import torch
import random
from PIL import Image, ImageDraw

from neurons.utils.image import image_to_tensor


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
    "REAL_IMAGE": image_to_tensor(Image.open(r"tests/images/img.jpg")),
    "REAL_IMAGE_LOW_INFERENCE": image_to_tensor(
        Image.open(r"tests/images/img_low_inference_steps.jpg")
    ),
}
