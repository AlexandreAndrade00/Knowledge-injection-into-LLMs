from transformers import TextGenerationPipeline
from typing import Callable, Iterable


class PLM:
    __model: TextGenerationPipeline
    __input_formatter: Callable[[str, str], object]

    def __init__(self, model: TextGenerationPipeline, formatter: Callable[[str, str], object]):
        self.__model = model
        self.__input_formatter = formatter

    def inference(self, model_input: str, context: str) -> str:
        formated_input = self.__input_formatter(model_input, context)

        return self.__model(formated_input, max_new_tokens=2048)[0]["generated_text"][-1]["content"]

    def benchmark(self, dataset_w_context: Iterable[dict[str, str]]) -> Iterable[str]:
        data = list(map(lambda x: self.__input_formatter(x["question"], x["context"]), dataset_w_context))

        for elem in data:
            yield self.__model(elem, max_new_tokens=2048)[0]["generated_text"][-1]
