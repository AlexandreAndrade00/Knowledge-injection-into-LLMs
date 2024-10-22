from typing import Optional
from transformers import pipeline
from plm.plm import PLM
from enum import Enum
import torch


class Device(Enum):
    CPU = "cpu"
    GPU = "cuda:0"


class PLMBuilder:
    model_id: Optional[str] = None
    device: Device = Device.GPU

    def select_model(self, model: str) -> None:
        self.model_id = model

    def select_device(self, device: Device) -> None:
        self.device = device

    def build(self) -> PLM:
        if self.model_id is None:
            raise AttributeError

        return PLM(pipeline("text-generation", model=self.model_id, model_kwargs={"torch_dtype": torch.bfloat16},
                            device=self.device.value))
