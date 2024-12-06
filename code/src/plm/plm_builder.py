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
    model_name: str
    device: Device = Device.GPU
    __input_formatter: Callable[[str, str], object]

    def select_model(self, model: str) -> None:
        self.model_id = model

    def select_device(self, device: Device) -> None:
        self.device = device

    def set_input_formatter(self, formatter: Callable[[str, str], object]):
        self.__input_formatter = formatter

    def set_model_name(self, name: str):
        self.model_name = name

    def build(self) -> PLM:
        if self.model_id is None:
            raise AttributeError

        pipe = pipeline(
            "text-generation",
            model=self.model_id,
            model_kwargs={"torch_dtype": torch.bfloat16},
            device=self.device.value,
        )

        pipe.tokenizer.pad_token_id = pipe.tokenizer.eos_token_id

        return PLM(pipe, self.__input_formatter, self.model_name)
