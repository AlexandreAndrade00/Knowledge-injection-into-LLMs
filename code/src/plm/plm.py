from transformers import TextGenerationPipeline


class PLM:
    __model: TextGenerationPipeline

    def __init__(self, model: TextGenerationPipeline):
        self.__model = model

    def inference(self, model_input: str) -> str:
        spans = self.__refined.process_text(model_input)

        messages = [
            {"role": "system",
             "content": "You are a chatbot with the objective of answering user questions concisely!"},
            {"role": "user", "content": model_input},
        ]

        return self.__model(messages, max_new_tokens=256)[0]["generated_text"][-1]["content"]
