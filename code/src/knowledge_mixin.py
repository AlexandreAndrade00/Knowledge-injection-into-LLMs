import json
from functools import reduce

from qwikidata.json_dump import WikidataJsonDump
from strsimpy import SorensenDice
from refined.inference.processor import Refined
from SPARQLWrapper import SPARQLWrapper, JSON
from enum import Enum
from typing import Optional, Any
from sentence_transformers import SentenceTransformer


class Type(Enum):
    ENTITY = 1
    PROPERTY = 2
    LITERAL = 3


class Entity:
    def __init__(self, id: str | None, label: str, type: Type):
        self.id = id
        self.label = label
        self.type = type


class Path:
    def __init__(self, entities: list[Entity], verbalized: str):
        self.entities = entities
        self.verbalized = verbalized


class KnowledgeMixin:
    __wikidata_sparql_endpoint: str = "https://query.wikidata.org/sparql"
    __wjd_path = "../../wikidata-20190401-all.json.bz2"
    __text_sim_model = SentenceTransformer("all-MiniLM-L6-v2", model_kwargs={"torch_dtype": "bfloat16"})

    def ranked_entities_statements(self, question: str, hops: int = 3) -> list[Path]:
        __refined: Refined = Refined.from_pretrained(model_name='wikipedia_model',
                                                     entity_set='wikipedia',
                                                     device='cuda:0',
                                                     use_precomputed_descriptions=True)

        spans = __refined.process_text(question)

        entities_to_query: list[Entity] = []
        entities_ids_visited: list[str] = []

        for span in spans:
            entities_to_query.append(Entity(span.predicted_entity.wikidata_entity_id,
                                            span.predicted_entity.wikipedia_entity_title, Type.ENTITY))

            entities_ids_visited.append(span.predicted_entity.wikidata_entity_id)

        ranked: list[list[Path]] = []

        for i in range(hops):
            entities_triples: list[list[Path]] = [
                self.get_entity_claims_sqarql(entity) for entity in entities_to_query
            ]

            entities_to_query.clear()

            ranked.append(self.k_similar_triples(question, entities_triples, k=1000))

            for triple in ranked[-1]:

                if triple.entities[-1].type == Type.ENTITY and triple.entities[-1].id not in entities_ids_visited:
                    entities_to_query.append(triple.entities[-1])
                    entities_ids_visited.append(triple.entities[-1].id)

                if triple.entities[0].type == Type.ENTITY and triple.entities[0].id not in entities_ids_visited:
                    entities_to_query.append(triple.entities[0])
                    entities_ids_visited.append(triple.entities[0].id)

        final_ranking = self.k_similar_triples(question, ranked, k=10)

        return final_ranking

    def k_similar_triples(self, question: str, verbalized_triples: list[list[Path]], **kwargs) -> list[Path]:
        k = kwargs.get('k', 10)

        ranked_triples: list[tuple[Path, float]] = []

        for entity_triples in verbalized_triples:
            ranked_triples.extend(self.__k_similar_triples_for_one_entity(question, entity_triples, 'sbert', k=k))

        ranked_triples.sort(key=lambda triple_ranking: triple_ranking[1])

        return [triple[0] for triple in ranked_triples[-k:]]

    def __k_similar_triples_for_one_entity(self, question: str, triples: list[Path], method, **kwargs) -> list[
        tuple[Path, float]]:
        k = kwargs.get('k', 10)

        if len(triples) == 0:
            return []

        if method == 'sorensen_dice':
            ranked_triples: list[tuple[Path, float]] = \
                [(triple, SorensenDice().similarity(question, triple.verbalized))
                 for triple in triples]
        else:
            embeddings1 = self.__text_sim_model.encode(question)
            embeddings2 = self.__text_sim_model.encode([triple.verbalized for triple in triples])

            similarities = self.__text_sim_model.similarity(embeddings1, embeddings2)

            ranked_triples: list[tuple[Path, float]] = []

            for i in range(len(triples)):
                ranked_triples.append((triples[i], similarities[0][i].item()))

        ranked_triples.sort(key=lambda triple_ranking: triple_ranking[1])

        return ranked_triples[-k:]

    def get_entity_claims_sqarql(self, entity: Entity) -> list[Path]:
        wikidataSPARQL = SPARQLWrapper(self.__wikidata_sparql_endpoint,
                                       agent="QAChatBot/0.1")

        wikidataSPARQL.setReturnFormat(JSON)

        wikidataSPARQL.setQuery(f"""
            SELECT ?prop1 ?prop1Label ?b ?bLabel ?c ?cLabel ?prop2 ?prop2Label ?d ?dLabel ?f ?fLabel
            WHERE
            {{
              {{
                wd:{entity.id} ?a ?b.
                ?b wdt:P31 ?c.
                ?prop1 wikibase:directClaim ?a .
                ?prop1 rdfs:label ?prop1Label.  filter(lang(?prop1Label) = "en").
              }}
              UNION
              {{
                ?d ?e wd:{entity.id}.
                ?d wdt:P31 ?f.
                ?prop2 wikibase:directClaim ?e .
                ?prop2 rdfs:label ?prop2Label.  filter(lang(?prop2Label) = "en").
              }}
            
              SERVICE wikibase:label {{ bd:serviceParam wikibase:language "en". }}
            }}
        """)

        result = wikidataSPARQL.queryAndConvert()['results']['bindings']

        triples: list[Path] = []

        for item in result:
            triple: Path

            if 'b' in item:
                prop_key = 'prop1'
                entity_key = 'b'
                entity_instance_key = 'c'
                direction = 'out'
            else:
                prop_key = 'prop2'
                entity_key = 'd'
                entity_instance_key = 'f'
                direction = 'in'

            property: Entity = Entity(item[prop_key]["value"].split('/')[-1], item[prop_key + "Label"]["value"],
                                      Type.PROPERTY)

            target_entity: Entity
            entity_instance: Optional[Entity] = None

            if item[entity_key]['type'] == 'uri':
                target_entity = Entity(item[entity_key]["value"].split('/')[-1], item[entity_key + "Label"]["value"],
                                       Type.ENTITY)

                entity_instance = Entity(item[entity_instance_key]["value"].split('/')[-1],
                                         item[entity_instance_key + "Label"]["value"],
                                         Type.ENTITY)
            elif item[entity_key]['type'] == 'literal':
                target_entity = Entity(None, item[entity_key + "Label"]["value"], Type.LITERAL)
            else:
                raise Exception("Unknown type")

            if direction == 'out':
                entities: list[Entity] = [entity, property, target_entity]

                if entity_instance is not None:
                    entities.insert(-2, entity_instance)
            elif direction == 'in':
                entities: list[Entity] = [target_entity, property, entity]

                if entity_instance is not None:
                    entities.insert(0, entity_instance)
            else:
                raise Exception("Unknown direction")

            triples.append(Path(entities, reduce(lambda s1, s2: f"{s1} {s2}", map(lambda e: e.label, entities), "")))

        return triples

    def get_entity_claims_local(self, entity: Entity) -> list[Path]:

        with open(self.__wjd_path, 'r') as fp:

            while True:
                entity_json = json.loads(fp.readline())

                if entity_json["id"] == entity.id:
                    break

            claims: dict[str, list] = entity_json["claims"]

            entities_to_explore: list[str] = []

            paths = list[Path]

            for claim_id, claim in claims.items():
                for snak_id, snak in claim["references"][0]['snaks'].items():
                    snak = snak[0]

                    if snak["datatype"] == "wikibase-item":
                        if snak["datavalue"]["type"] == "wikibase-entityid":
                            entities_to_explore.append(snak["datavalue"]["value"]["id"])
                    elif snak["datatype"] == "string":
                        paths.append(Path([entity, Entity(snak_id, '', Type.PROPERTY), Entity(null, )], ''))

        return []


def main():
    result = KnowledgeMixin().ranked_entities_statements('How many episodes have The Walking Dead?', hops=2)

    print(result)


if __name__ == '__main__':
    main()
