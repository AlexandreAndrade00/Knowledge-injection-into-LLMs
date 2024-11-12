from typing import Optional
from transformers import pipeline
from plm.plm import PLM
from enum import Enum
from typing import Callable
import torch


class Device(Enum):
    CPU = "cpu"
    GPU = "cuda:0"


class PLMBuilder:
    model_id: Optional[str] = None
    device: Device = Device.GPU
    __input_formatter: Callable[[str, str], object]

    def select_model(self, model: str) -> None:
        self.model_id = model

    def select_device(self, device: Device) -> None:
        self.device = device

    def set_input_formatter(self, formatter: Callable[[str, str], object]):
        self.__input_formatter = formatter

    def build(self) -> PLM:
        if self.model_id is None:
            raise AttributeError

        return PLM(pipeline("text-generation", model=self.model_id, model_kwargs={"torch_dtype": torch.bfloat16},
                            device=self.device.value), self.__input_formatter)
