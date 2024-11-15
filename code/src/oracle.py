from plm.plm import PLM
from functools import reduce
from knowledge_mixin import KnowledgeMixin


class Oracle(KnowledgeMixin):
    __model: PLM
    __question: str

    def __init__(self, model: PLM):
        self.__model = model

    def set_question(self, question: str):
        self.__question = question

    def answer(self) -> str:
        context_list: list[str] = super().ranked_entities_statements(self.__question)

        context: str = reduce(lambda s1, s2: f"{s1}\n{s2}\n", context_list) if len(context_list) > 0 else ""

        return self.__model.inference(self.__question, context)
