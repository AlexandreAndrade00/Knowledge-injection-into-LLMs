from transformers import TextGenerationPipeline


class PLM:
    _model: TextGenerationPipeline

    def __init__(self, model: TextGenerationPipeline):
        self._model = model

    def inference(self, model_input: str) -> str:
        return self._model(model_input, max_length=100)
