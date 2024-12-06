from functools import reduce
from strsimpy import SorensenDice
from refined.inference.processor import Refined
from SPARQLWrapper import SPARQLWrapper, JSON
from enum import Enum
from typing import Optional, Any
from sentence_transformers import SentenceTransformer
import orjson


class Type(Enum):
    ENTITY = 1
    PROPERTY = 2
    LITERAL = 3


class DataOrigin(Enum):
    SPARQL = 1
    LOCAL = 2


class Entity:
    def __init__(self, id: str | None, label: str, type: Type):
        self.id = id
        self.label = label
        self.type = type

    def __str__(self) -> str:
        return f"{{id: {self.id}, label: {self.label}, type: {self.type.name}}}"


class Path:
    def __init__(self, entities: list[Entity], verbalized: str):
        self.entities = entities
        self.verbalized = verbalized


class KnowledgeMixin:
    __wikidata_sparql_endpoint: str = "https://query.wikidata.org/sparql"
    __wjd_path = "../../data/wikidata-20240101-all.json"
    __mapping_path = "../../data/entities_mapping_bytes.json"
    __text_sim_model = SentenceTransformer(
        "all-MiniLM-L6-v2", model_kwargs={"torch_dtype": "bfloat16"}
    )
    __entity_cache: dict[str, list[Path]] = {}
    __refined: Refined = Refined.from_pretrained(
        model_name="wikipedia_model",
        entity_set="wikipedia",
        device="cuda:0",
        use_precomputed_descriptions=True,
    )

    def __init__(self, data_origin: DataOrigin):
        self.__data_origin = data_origin

        if data_origin.name == DataOrigin.LOCAL.name:
            self.__mapping = orjson.loads(open(self.__mapping_path).read())

    def ranked_entities_statements(
        self, question: str, hops: int, k: int
    ) -> list[Path]:

        spans = self.__refined.process_text(question)

        entities_to_query: list[Entity] = []
        entities_ids_visited: list[str] = []

        for span in spans:
            entities_to_query.append(
                Entity(
                    span.predicted_entity.wikidata_entity_id,
                    span.predicted_entity.wikipedia_entity_title,
                    Type.ENTITY,
                )
            )

            entities_ids_visited.append(span.predicted_entity.wikidata_entity_id)

        ranked: list[list[Path]] = []

        for i in range(hops):
            entities_triples: list[list[Path]] = [
                (
                    self.get_entity_claims_sqarql(entity)
                    if self.__data_origin == DataOrigin.SPARQL
                    else self.get_entity_claims_local(entity)
                )
                for entity in entities_to_query
            ]

            entities_to_query.clear()

            ranked.append(self.k_similar_triples(question, entities_triples, k=1000))

            for triple in ranked[-1]:
                if (
                    triple.entities[-1].type.name == Type.ENTITY.name
                    and triple.entities[-1].id not in entities_ids_visited
                ):
                    if triple.entities[-1].id is None:
                        print(triple.entities[-1])
                        continue

                    entities_to_query.append(triple.entities[-1])
                    entities_ids_visited.append(triple.entities[-1].id)

                if (
                    triple.entities[0].type.name == Type.ENTITY.name
                    and triple.entities[0].id not in entities_ids_visited
                ):
                    if triple.entities[0].id is None:
                        print(triple.entities[0])
                        continue

                    entities_to_query.append(triple.entities[0])
                    entities_ids_visited.append(triple.entities[0].id)

        final_ranking = self.k_similar_triples(question, ranked, k=k)

        return final_ranking

    def k_similar_triples(
        self, question: str, verbalized_triples: list[list[Path]], **kwargs
    ) -> list[Path]:
        k = kwargs.get("k", 10)

        ranked_triples: list[tuple[Path, float]] = []

        for entity_triples in verbalized_triples:
            ranked_triples.extend(
                self.__k_similar_triples_for_one_entity(
                    question, entity_triples, "sbert", k=k
                )
            )

        ranked_triples.sort(key=lambda triple_ranking: triple_ranking[1])

        return [triple[0] for triple in ranked_triples[-k:]]

    def __k_similar_triples_for_one_entity(
        self, question: str, triples: list[Path], method, **kwargs
    ) -> list[tuple[Path, float]]:
        k = kwargs.get("k", 10)

        if len(triples) == 0:
            return []

        if method == "sorensen_dice":
            ranked_triples: list[tuple[Path, float]] = [
                (triple, SorensenDice().similarity(question, triple.verbalized))
                for triple in triples
            ]
        else:
            embeddings1 = self.__text_sim_model.encode(question)
            embeddings2 = self.__text_sim_model.encode(
                [triple.verbalized for triple in triples]
            )

            similarities = self.__text_sim_model.similarity(embeddings1, embeddings2)

            ranked_triples: list[tuple[Path, float]] = []

            for i in range(len(triples)):
                ranked_triples.append((triples[i], similarities[0][i].item()))

        ranked_triples.sort(key=lambda triple_ranking: triple_ranking[1])

        return ranked_triples[-k:]

    def get_entity_claims_sqarql(self, entity: Entity) -> list[Path]:
        if entity.id in self.__entity_cache:
            return self.__entity_cache[entity.id]

        wikidataSPARQL = SPARQLWrapper(
            self.__wikidata_sparql_endpoint, agent="QAChatBot/0.1"
        )

        wikidataSPARQL.setReturnFormat(JSON)

        wikidataSPARQL.setQuery(
            f"""
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
        """
        )

        result = wikidataSPARQL.queryAndConvert()["results"]["bindings"]

        triples: list[Path] = []

        for item in result:
            triple: Path

            if "b" in item:
                prop_key = "prop1"
                entity_key = "b"
                entity_instance_key = "c"
                direction = "out"
            else:
                prop_key = "prop2"
                entity_key = "d"
                entity_instance_key = "f"
                direction = "in"

            property: Entity = Entity(
                item[prop_key]["value"].split("/")[-1],
                item[prop_key + "Label"]["value"],
                Type.PROPERTY,
            )

            target_entity: Entity
            entity_instance: Optional[Entity] = None

            if item[entity_key]["type"] == "uri":
                target_entity = Entity(
                    item[entity_key]["value"].split("/")[-1],
                    item[entity_key + "Label"]["value"],
                    Type.ENTITY,
                )

                entity_instance = Entity(
                    item[entity_instance_key]["value"].split("/")[-1],
                    item[entity_instance_key + "Label"]["value"],
                    Type.ENTITY,
                )
            elif item[entity_key]["type"] == "literal":
                target_entity = Entity(
                    None, item[entity_key + "Label"]["value"], Type.LITERAL
                )
            else:
                raise Exception("Unknown type")

            if direction == "out":
                entities: list[Entity] = [entity, property, target_entity]

                if entity_instance is not None:
                    entities.insert(-2, entity_instance)
            elif direction == "in":
                entities: list[Entity] = [target_entity, property, entity]

                if entity_instance is not None:
                    entities.insert(0, entity_instance)
            else:
                raise Exception("Unknown direction")

            triples.append(
                Path(
                    entities,
                    reduce(
                        lambda s1, s2: f"{s1} {s2}",
                        map(lambda e: e.label, entities),
                        "",
                    ),
                )
            )

        self.__entity_cache[entity.id] = triples

        return triples

    def get_entity_claims_local(self, entity: Entity) -> list[Path]:

        if entity.id is None:
            print(entity)
            return []

        if entity.id in self.__entity_cache:
            return self.__entity_cache[entity.id]

        entity_json = self.get_entity_json_by_id(entity.id)

        if entity_json is None:
            print(entity)
            return []

        claims: dict[str, list] = entity_json["claims"]

        paths: list[Path] = []

        for property_id, snaks in claims.items():
            for snak in snaks:
                if snak["rank"] == "deprecated":
                    continue

                main_snack = snak["mainsnak"]

                if main_snack["snaktype"] != "value":
                    continue

                property: Entity = self.get_entity_by_id(property_id)

                if property is None:
                    print(property_id)
                    continue

                data_value = main_snack["datavalue"]

                snak_entity: Entity

                if data_value["type"] == "string":
                    snak_entity = Entity(None, data_value["value"], Type.LITERAL)
                elif data_value["type"] == "wikibase-entityid":
                    snak_entity = self.get_entity_by_id(data_value["value"]["id"])

                    if snak_entity is None:
                        print(data_value["value"]["id"])
                        continue
                elif data_value["type"] == "quantity":
                    snak_entity = Entity(
                        None, data_value["value"]["amount"], Type.LITERAL
                    )
                elif data_value["type"] == "monolingualtext":
                    if data_value["value"]["language"] != "en":
                        continue

                    snak_entity = Entity(
                        None, data_value["value"]["text"], Type.LITERAL
                    )
                elif data_value["type"] == "globecoordinate":
                    snak_entity = Entity(
                        None,
                        f"({data_value["value"]["latitude"]}, {data_value["value"]["longitude"]})",
                        Type.LITERAL,
                    )
                elif data_value["type"] == "time":
                    snak_entity = Entity(
                        None, data_value["value"]["time"], Type.LITERAL
                    )
                else:
                    raise Exception(f'Unknown type "{data_value["type"]}"')

                triple = [entity, property, snak_entity]

                paths.append(
                    Path(
                        triple,
                        reduce(
                            lambda s1, s2: f"{s1} {s2}",
                            map(lambda e: e.label, triple),
                            "",
                        ),
                    )
                )

        self.__entity_cache[entity.id] = paths

        return paths

    def get_entity_by_id(self, entity_id: str) -> Optional[Entity]:
        entity_json = self.get_entity_json_by_id(entity_id)

        if entity_json is None:
            return None

        labels: dict[str, dict[str, str]] = entity_json["labels"]

        label: str

        if "en" in labels:
            label = labels["en"]["value"]
        elif "en-gb" in labels:
            label = labels["en-gb"]["value"]
        elif "en-us" in labels:
            label = labels["en-us"]["value"]
        else:
            label = next(iter(labels.values()), {"value": ""})["value"]

        entity = Entity(
            entity_json["id"],
            label,
            Type.ENTITY if entity_json["id"].startswith("Q") else Type.PROPERTY,
        )

        return entity

    def get_entity_json_by_id(self, entity_id) -> Optional[dict[str, Any]]:
        if not entity_id in self.__mapping:
            return None

        with open(self.__wjd_path, "rt") as fp:
            fp.seek(self.__mapping[entity_id])

            entity_json_str = fp.readline()

            entity_json = orjson.loads(entity_json_str[:-2])

        return entity_json


def main():
    result = KnowledgeMixin(DataOrigin.LOCAL).ranked_entities_statements(
        "How many episodes have The Walking Dead?", hops=1
    )

    print(result)


if __name__ == "__main__":
    main()
