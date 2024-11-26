from plm.plm import PLM
from functools import reduce
from knowledge_mixin import KnowledgeMixin, Path


class Oracle(KnowledgeMixin):
    __model: PLM
    __question: str
    __use_context: bool = True

    def __init__(self, model: PLM, use_context: bool):
        self.__model = model
        self.__use_context = use_context

    def set_question(self, question: str):
        self.__question = question

    def answer(self) -> str:
        context: str = ''

        if self.__use_context:
            context_list: list[Path] = super().ranked_entities_statements(self.__question)

            context = reduce(lambda s1, s2: f"{s1}\n{s2}", set(map(lambda e: e.verbalized, context_list))) if len(
                context_list) > 0 else ""

        return self.__model.inference(self.__question, context)
