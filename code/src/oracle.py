from refined.inference.processor import Refined
from plm.plm import PLM


class Oracle:
    __model: PLM
    __refined: Refined

    def __init__(self, model: PLM):
        self.__model = model
        self.__refined = Refined.from_pretrained(model_name='wikipedia_model_with_numbers',
                                                 entity_set='wikipedia',
                                                 use_precomputed_descriptions=True)

    def answer(self, model_input: str) -> str:
        spans = self.__refined.process_text(model_input)

        return self.__model.inference(model_input)
