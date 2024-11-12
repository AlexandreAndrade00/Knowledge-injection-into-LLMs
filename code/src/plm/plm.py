from transformers import TextGenerationPipeline
from typing import Callable


class PLM:
    __model: TextGenerationPipeline
    __input_formatter: Callable[[str, str], object]

    def __init__(self, model: TextGenerationPipeline, formatter: Callable[[str, str], object]):
        self.__model = model
        self.__input_formatter = formatter

    def inference(self, model_input: str, context: str) -> str:
        formated_input = self.__input_formatter(model_input, context)

        return self.__model(formated_input, max_new_tokens=256)[0]["generated_text"][-1]["content"]
