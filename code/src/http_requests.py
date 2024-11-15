from SPARQLWrapper import SPARQLWrapper, JSON
from wikidata.client import Client
from wikidata.entity import EntityId, Entity
from wikidata.quantity import Quantity
from datetime import date


class HttpRequests:
    wikidata_sparql_endpoint: str = "https://query.wikidata.org/sparql"

    def get_entity_claims(self, entity_id: str, entity_name: str) -> list:
        wikidataSPARQL = SPARQLWrapper(self.wikidata_sparql_endpoint,
                                       agent="QAChatBot/0.1")

        wikidataSPARQL.setReturnFormat(JSON)

        wikidataSPARQL.setQuery(f"""
            SELECT ?propLabel ?bLabel
            WHERE
            {{
              wd:{entity_id} ?a ?b.
            
              SERVICE wikibase:label {{ bd:serviceParam wikibase:language "en". }} 
              ?prop wikibase:directClaim ?a .
              ?prop rdfs:label ?propLabel.  filter(lang(?propLabel) = "en").
            }}
        """)

        result = wikidataSPARQL.queryAndConvert()['results']['bindings']

        verbalized_claims = []

        for item in result:
            verbalized_claims.append(f"{entity_name} {item["propLabel"]["value"]} {item["bLabel"]["value"]}")

        return verbalized_claims
