from strsimpy import SorensenDice
from refined.inference.processor import Refined
from SPARQLWrapper import SPARQLWrapper, JSON
from enum import Enum


class Type(Enum):
    ENTITY = 1
    PROPERTY = 2


class Entity:
    def __init__(self, id: str, label: str, type: Type):
        self.id = id
        self.label = label
        self.type = type


class Triple:
    def __init__(self, entities: list[Entity], verbalized: str):
        self.entities = entities
        self.verbalized = verbalized


class KnowledgeMixin:
    __refined: Refined = Refined.from_pretrained(model_name='wikipedia_model',
                                                 entity_set='wikipedia',
                                                 device='cpu',
                                                 use_precomputed_descriptions=True)

    __wikidata_sparql_endpoint: str = "https://query.wikidata.org/sparql"

    def ranked_entities_statements(self, question: str) -> list[Triple]:
        spans = self.__refined.process_text(question)

        entities_triples: list[list[Triple]] = [
            self.get_entity_claims(
                span.predicted_entity.wikidata_entity_id,
                span.predicted_entity.wikipedia_entity_title
            ) for span in spans
        ]

        ranked = self.k_similar_triples(question, entities_triples)

        return ranked

    def k_similar_triples(self, question: str, verbalized_triples: list[list[Triple]], **kwargs) -> list[Triple]:
        k = kwargs.get('k', 10)

        ranked_triples: list[tuple[Triple, float]] = []

        for entity_triples in verbalized_triples:
            ranked_triples.extend(self.__k_similar_triples_for_one_entity(question, entity_triples))

        ranked_triples.sort(key=lambda triple_ranking: triple_ranking[1])

        return [triple[0] for triple in ranked_triples[-k:]]

    @staticmethod
    def __k_similar_triples_for_one_entity(question: str, triples: list[Triple], **kwargs) -> list[
        tuple[Triple, float]]:
        k = kwargs.get('k', 10)

        # sbert
        ranked_triples: list[tuple[Triple, float]] = \
            [(triple, SorensenDice().similarity(question, triple.verbalized))
             for triple in triples]

        ranked_triples.sort(key=lambda triple_ranking: triple_ranking[1])

        return ranked_triples[-k:]

    def get_entity_claims(self, entity_id: str, entity_name: str) -> list[Triple]:
        wikidataSPARQL = SPARQLWrapper(self.__wikidata_sparql_endpoint,
                                       agent="QAChatBot/0.1")

        wikidataSPARQL.setReturnFormat(JSON)

        wikidataSPARQL.setQuery(f"""
            SELECT ?prop ?propLabel ?b ?bLabel
            WHERE
            {{
              wd:{entity_id} ?a ?b.

              SERVICE wikibase:label {{ bd:serviceParam wikibase:language "en". }} 
              ?prop wikibase:directClaim ?a .
              ?prop rdfs:label ?propLabel.  filter(lang(?propLabel) = "en").
            }}
        """)

        result = wikidataSPARQL.queryAndConvert()['results']['bindings']

        triples: list[Triple] = []

        for item in result:
            triple = Triple([Entity(entity_id, entity_name, Type.ENTITY),
                             Entity(item["prop"]["value"], item["propLabel"]["value"], Type.PROPERTY),
                             Entity(item["b"]["value"], item["bLabel"]["value"], Type.ENTITY)],
                            f"{entity_name} {item["propLabel"]["value"]} {item["bLabel"]["value"]}")

            triples.append(triple)

        return triples


def main():
    result = KnowledgeMixin().ranked_entities_statements('what is the capital of spain')

    print(result)


if __name__ == '__main__':
    main()
