from refined.inference.processor import Refined
from plm.plm import PLM
from http_requests import HttpRequests
from strsimpy import SorensenDice


class Oracle:
    __model: PLM
    __refined: Refined
    __http_requests: HttpRequests = HttpRequests()
    __question: str

    def __init__(self, model: PLM):
        self.__model = model
        self.__refined = Refined.from_pretrained(model_name='wikipedia_model',
                                                 entity_set='wikipedia',
                                                 use_precomputed_descriptions=True)

    def set_question(self, question: str):
        self.__question = question

    def answer(self) -> str:
        return self.__model.inference(self.__question)

    def __ranked_entities_statements(self) -> list[str]:
        spans = self.__refined.process_text(self.__question)

        verbalized_rdf = [self.__http_requests.get_entity_claims(span.predicted_entity.wikidata_entity_id) for span in
                          spans]

        return self.__k_similar_triples(verbalized_rdf)

    def __k_similar_triples(self, verbalized_triples: list[list[str]], **kwargs) -> list[str]:
        k = kwargs.get('k', 10)

        ranked_triples: list[tuple[str, float]] = []

        for entity_triples in verbalized_triples:
            ranked_triples.extend(self.__k_similar_triples_for_one_entity(entity_triples))

        ranked_triples.sort(key=lambda triple_ranking: triple_ranking[1])

        return [triple[0] for triple in ranked_triples[-k:]]

    def __k_similar_triples_for_one_entity(self, verbalized_triples: list[str], **kwargs) -> list[
        tuple[str, float]]:
        k = kwargs.get('k', 10)

        ranked_triples: list[tuple[str, float]] = [(triple, SorensenDice().similarity(self.__question, triple)) for
                                                   triple
                                                   in
                                                   verbalized_triples]

        ranked_triples.sort(key=lambda triple_ranking: triple_ranking[1])

        return ranked_triples[-k:]
