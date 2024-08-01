from pydantic import BaseModel
from typing import Optional, List
import torch


class NeuronAttributes(BaseModel):
    background_steps: Optional[int] = None
    total_number_of_neurons: int
    wallet_hotkey_ss58_address: Optional[str] = None
    hotkeys: Optional[List[str]] = None
    device: Optional[torch.device] = None

    class Config:
        arbitrary_types_allowed = True
