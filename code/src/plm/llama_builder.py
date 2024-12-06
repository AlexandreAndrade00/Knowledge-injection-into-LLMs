from __future__ import annotations

from plm.plm_builder import PLMBuilder, Device
from plm.plm import PLM


class LlamaBuilder:
    __params: int = 1
    __builder = PLMBuilder()

    def set_params(self, params: int) -> LlamaBuilder:
        self.__params = params

        return self

    def build(self) -> PLM:
        self.__builder.select_model(f"meta-llama/Llama-3.2-{self.__params}B-Instruct")

        self.__builder.set_model_name(f"llama{self.__params}b")

        self.__builder.select_device(Device.GPU)

        self.__builder.set_input_formatter(
            lambda model_input, context: (
                [
                    {
                        "role": "system",
                        "content": "You are a chatbot with the objective of answering user questions concisely and providing facts!",
                    },
                    {
                        "role": "system",
                        "content": f"The following sentences are obtained from up-to-date knowledge graphs, use them as factual information to answer the questions:\n{context}",
                    },
                    {"role": "user", "content": model_input},
                ]
                if context != ""
                else [
                    {
                        "role": "system",
                        "content": "You are a chatbot with the objective of answering user questions concisely and providing facts!",
                    },
                    {"role": "user", "content": model_input},
                ]
            )
        )

        return self.__builder.build()
