from enum import Enum
from typing import Any, Dict

from pydantic import BaseModel

from diffusers import AutoPipelineForText2Image


class TaskType(str, Enum):
    TEXT_TO_IMAGE = "TEXT_TO_IMAGE"
    IMAGE_TO_IMAGE = "IMAGE_TO_IMAGE"


class ModelConfig(BaseModel):
    args: Dict[str, Any]
    model: AutoPipelineForText2Image
