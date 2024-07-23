import copy
import time
import traceback
from typing import Dict, List, Any
import torchvision.transforms as transforms
import bittensor as bt
import torch
from loguru import logger
from neurons.miners.StableMiner.schema import ModelConfig, TaskType
from neurons.protocol import ImageGeneration
from neurons.utils.image import empty_image_tensor
from neurons.utils.nsfw import clean_nsfw_from_prompt
from neurons.utils.log import sh


def without_keys(d: Dict, keys: List[str]) -> Dict:
    return {k: v for k, v in d.items() if k not in keys}


def setup_model_args(
    synapse: ImageGeneration, model_config: ModelConfig
) -> Dict[str, Any]:
    model_args: Dict[str, Any] = copy.deepcopy(model_config.args)
    set_common_model_args(model_args, synapse)
    if synapse.generation_type.upper() == TaskType.IMAGE_TO_IMAGE:
        set_image_to_image_args(model_args, synapse)
    return model_args


def set_common_model_args(
    model_args: Dict[str, Any], synapse: ImageGeneration
) -> None:
    model_args["prompt"] = [clean_nsfw_from_prompt(synapse.prompt)]
    model_args["width"] = synapse.width
    model_args["height"] = synapse.height
    model_args["num_images_per_prompt"] = synapse.num_images_per_prompt
    model_args["guidance_scale"] = synapse.guidance_scale
    model_args["denoising_end"] = 0.8
    model_args["output_type"] = "latent"
    model_args["num_inference_steps"] = getattr(
        synapse, "steps", model_args.get("num_inference_steps", 50)
    )
    if synapse.negative_prompt:
        model_args["negative_prompt"] = [synapse.negative_prompt]


def set_image_to_image_args(
    model_args: Dict[str, Any], synapse: ImageGeneration
) -> None:
    model_args["image"] = transforms.transforms.ToPILImage()(
        bt.Tensor.deserialize(synapse.prompt_image)
    )


def setup_refiner_args(model_args: Dict[str, Any]) -> Dict[str, Any]:
    refiner_args = {
        "denoising_start": model_args["denoising_end"],
        "prompt": model_args["prompt"],
        "num_inference_steps": int(model_args["num_inference_steps"] * 0.2),
    }
    model_args["num_inference_steps"] = int(
        model_args["num_inference_steps"] * 0.8
    )
    return refiner_args


def filter_nsfw_images(
    images: List[torch.Tensor], nsfw_image_filter_func
) -> List[torch.Tensor]:
    if not images:
        return images
    try:
        if any(nsfw_image_filter_func(images)):
            logger.info("An image was flagged as NSFW: discarding image.")
            return [empty_image_tensor() for _ in images]
    except Exception as e:
        traceback.print_exc()
        logger.error(f"Error in NSFW filtering: {e}")
    return images


def log_generation_time(start_time: float, total_requests: int) -> None:
    generation_time: float = time.perf_counter() - start_time
    total_requests += 1
    average_time: float = generation_time / total_requests
    logger.info(
        f"{sh('Time')} -> {generation_time:.2f}s | Average: {average_time:.2f}s"
    )


def log_gpu_memory_usage(stage: str) -> None:
    try:
        allocated = torch.cuda.memory_allocated() / 1024**2
        max_allocated = torch.cuda.max_memory_allocated() / 1024**2
        total = torch.cuda.get_device_properties(0).total_memory / 1024**2
        free = total - allocated

        logger.info(f"GPU memory allocated {stage}: {allocated:.2f} MB")
        logger.info(f"Max GPU memory allocated {stage}: {max_allocated:.2f} MB")
        logger.info(f"Total GPU memory: {total:.2f} MB")
        logger.info(f"Free GPU memory: {free:.2f} MB")
    except Exception as e:
        logger.error(f"Failed to log GPU memory usage {stage}: {str(e)}")
