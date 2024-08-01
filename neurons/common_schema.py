from pydantic import BaseModel
from typing import Optional, List
import torch


class NeuronAttributes(BaseModel):
    background_steps: int
    total_number_of_neurons: int
    wallet_hotkey_ss58_address: str
    hotkeys: List[str]
    device: Optional[torch.device] = None

    class Config:
        arbitrary_types_allowed = True