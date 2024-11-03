import requests
from rdflib import Graph


class HttpRequests:
    __wikidata_entity_endpoint_url: str = 'https://www.wikidata.org/wiki/Special:EntityData'
    __wikidata_sparql_endpoint_url: str = 'https://query.wikidata.org/sparql'

    def get_entity_statements(self, entity_id: str) -> list:
        payload: dict[str, str] = {
            'query': f'DESCRIBE wd:{entity_id}',
            'format': 'json'
        }

        response = requests.get(self.__wikidata_sparql_endpoint_url, params=payload)

        results = response.json()

        triples: list = []
        for result in results["results"]["bindings"]:
            triples.append((result["subject"], result["predicate"], result["object"]))

        return triples
