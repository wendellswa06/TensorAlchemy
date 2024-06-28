import torch
import torchvision.transforms as transforms
from diffusers import DiffusionPipeline, StableDiffusionXLImg2ImgPipeline
from diffusers.utils import load_image
from transformers import CLIPImageProcessor

transform = transforms.Compose([transforms.PILToTensor()])

pipe = DiffusionPipeline.from_pretrained(
    "stabilityai/stable-diffusion-xl-base-1.0",
    torch_dtype=torch.float16,
    use_safetensors=True,
    variant="fp16",
)
pipe.to("cuda")

prompt = "An astronaut riding a green horse"
args = {
    "prompt": prompt,
    "guidance_scale": 7.5,
    "num_inference_steps": 20,
    "denoising_end": 0.8,
    "output_type": "latent",
}

images = pipe(**args).images[0]

processor = CLIPImageProcessor()

clip_input = processor(
    [image for image in [images]],
    return_tensors="pt",
)

args = {
    "prompt": prompt,
    "guidance_scale": 7.5,
    "num_inference_steps": 20,
    "denoising_end": 0.8,
    # "output_type": "latent",
}

images_pil = transform(pipe(**args).images[0])

clip_input = processor(
    [image for image in [images_pil]],
    return_tensors="pt",
)

refiner = DiffusionPipeline.from_pretrained(
    "stabilityai/stable-diffusion-xl-refiner-1.0",
    text_encoder_2=pipe.text_encoder_2,
    vae=pipe.vae,
    torch_dtype=torch.float16,
    use_safetensors=True,
    variant="fp16",
)
refiner.to("cuda")
