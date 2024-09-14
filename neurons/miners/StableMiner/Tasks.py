import asyncio
import io
import os
import time
from taskiq_redis import ListQueueBroker, RedisAsyncResultBackend
import base64
from diffusers import (
    DiffusionPipeline,
    StableDiffusionXLPipeline, 
    KDPM2AncestralDiscreteScheduler,
    EulerAncestralDiscreteScheduler,
    AutoencoderKL,
    AutoPipelineForImage2Image,
    AutoPipelineForText2Image,
    DPMSolverMultistepScheduler,
)
from PIL import Image
import torch
import ImageReward as reward
import random
import redis
import argparse
from transformers import CLIPImageProcessor, CLIPProcessor, CLIPModel
import pathlib, sys
from neurons.utils.image import image_to_base64

cuda_device_id = os.environ.get('CUDA_VISIBLE_DEVICES', 'Not')
if cuda_device_id != 'Not':
    cuda_device_id = int(cuda_device_id)

redis_async_result = RedisAsyncResultBackend(
    redis_url="redis://localhost:6379",
)

# Or you can use PubSubBroker if you need broadcasting
broker = ListQueueBroker(
    url="redis://localhost:6379",
    result_backend=redis_async_result,
)

result = []

negative_prompt = "nsfw, bad quality, bad anatomy, worst quality, low quality, low resolutions, extra fingers, blur, blurry, ugly, wrongs proportions, watermark, image artifacts, lowres, ugly, jpeg artifacts, deformed, noisy image"

## proteus
# Load VAE component
vae = AutoencoderKL.from_pretrained(
    "madebyollin/sdxl-vae-fp16-fix", 
    torch_dtype=torch.float16
)
proteus_pipe = StableDiffusionXLPipeline.from_pretrained(
    "dataautogpt3/ProteusV0.4", 
    vae=vae,
    torch_dtype=torch.float16
)

proteus_pipe.scheduler = KDPM2AncestralDiscreteScheduler.from_config(proteus_pipe.scheduler.config)

## playground
playground_pipe = DiffusionPipeline.from_pretrained("playgroundai/playground-v2.5-1024px-aesthetic", torch_dtype=torch.float16, variant="fp16")

## juggernautxl
juggernautxl_pipe = DiffusionPipeline.from_pretrained("RunDiffusion/Juggernaut-XL-v9", torch_dtype=torch.float16, use_safetensors=True, variant="fp16")

## zavychroma
zavychroma_pipe = DiffusionPipeline.from_pretrained("imagepipeline/Zavy-Chroma-XL-v3", torch_dtype=torch.float16, use_safetensors=True, variant="fp16")

sdxl_names = ["proteus", "playground", "juggernautxl", "zavychroma"]
pipes = [proteus_pipe, playground_pipe, juggernautxl_pipe, zavychroma_pipe]
pipe = pipes[cuda_device_id % 4]
print(f"Model name is {sdxl_names[cuda_device_id]}\n")

# ImageReward model.
scoring_model = reward.load("ImageReward-v1.0")

# Clip model
clip_model = CLIPModel.from_pretrained( "openai/clip-vit-base-patch32" )
clip_processor = CLIPProcessor.from_pretrained( "openai/clip-vit-base-patch32" )
threshold = 0.0

@broker.task
async def generate_image(prompt: str, guidance_scale: float, num_inference_steps: int):
    start_time = time.time()
    pipe.to("cuda")
    
    global result
    
    """Solve all problems in the world."""
    
    print(f"-------------prompt in broker: {prompt}-------------------")
    print(f"-------------Guidance_scale in broker: {guidance_scale}-------------------")
    
    images = pipe(prompt=prompt, negative_prompt=negative_prompt, width=1024, height=1024, num_inference_steps=30, guidance_scale=7.5).images
    # Playground
    # images = playground_pipe(prompt=prompt, negative_prompt=negative_prompt, width=1024, height=1024, num_inference_steps=35, guidance_scale=3).images
    
    end_time = time.time()

    print(f"Successfully generated images in {end_time-start_time} seconds.")
    
    score = scoring_model.score(prompt, images)
    # Note: Xrunner: clip model evaluate
    inputs = clip_processor(text=[prompt], images=images[0], return_tensors="pt", padding=True)
    outputs = clip_model(**inputs)
    logit = outputs.logits_per_image.squeeze().tolist()
    # images[0].save(f"{score}-{prompt}.png")
    # Note: Xrunner: encode <class 'PIL.Image.Image'>
    base64_image = image_to_base64(images[0])
    print("All problems are solved!")
    # return images, score
    return {"prompt": prompt, "score": score, "logit": logit, "image": base64_image}

if __name__ == "__main__":
    # Add the base repository to the path so the miner can access it
    file_path = str(
        pathlib.Path(__file__).parent.parent.parent.parent.resolve(),
    )
    if file_path not in sys.path:
        sys.path.append(file_path)

    # Import StableMiner after fixing path
    from stable_miner import StableMiner
    broker.startup()
    # Start the miner
    StableMiner()
    