from strsimpy import SorensenDice
from http_requests import HttpRequests
from refined.inference.processor import Refined


class KnowledgeMixin:
    __refined: Refined = Refined.from_pretrained(model_name='wikipedia_model',
                                                 entity_set='wikipedia',
                                                 device='cuda:0',
                                                 use_precomputed_descriptions=True)

    __http_requests: HttpRequests = HttpRequests()

    def ranked_entities_statements(self, question: str) -> list[str]:
        spans = self.__refined.process_text(question)

        verbalized_rdf = [
            self.__http_requests.get_entity_claims(
                span.predicted_entity.wikidata_entity_id,
                span.predicted_entity.wikipedia_entity_title
            ) for span in spans
        ]

        return self.__k_similar_triples(question, verbalized_rdf)

    def __k_similar_triples(self, question: str, verbalized_triples: list[list[str]], **kwargs) -> list[str]:
        k = kwargs.get('k', 10)

        ranked_triples: list[tuple[str, float]] = []

        for entity_triples in verbalized_triples:
            ranked_triples.extend(self.__k_similar_triples_for_one_entity(question, entity_triples))

        ranked_triples.sort(key=lambda triple_ranking: triple_ranking[1])

        return [triple[0] for triple in ranked_triples[-k:]]

    @staticmethod
    def __k_similar_triples_for_one_entity(question: str, verbalized_triples: list[str], **kwargs) -> list[
        tuple[str, float]]:
        k = kwargs.get('k', 10)

        # sbert
        ranked_triples: list[tuple[str, float]] = [(triple, SorensenDice().similarity(question, triple)) for
                                                   triple
                                                   in
                                                   verbalized_triples]

        ranked_triples.sort(key=lambda triple_ranking: triple_ranking[1])

        return ranked_triples[-k:]
