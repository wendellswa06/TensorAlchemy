from enum import Enum
from typing import Any, Dict, Optional

from diffusers import DiffusionPipeline
from pydantic import BaseModel


class TaskType(str, Enum):
    TEXT_TO_IMAGE = "TEXT_TO_IMAGE"
    IMAGE_TO_IMAGE = "IMAGE_TO_IMAGE"


class ModelConfig(BaseModel):
    args: Dict[str, Any]
    model: DiffusionPipeline
    refiner: Optional[DiffusionPipeline]

    class Config:
        arbitrary_types_allowed = True
