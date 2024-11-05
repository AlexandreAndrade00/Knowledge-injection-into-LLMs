from wikidata.client import Client
from wikidata.entity import EntityId, Entity
from wikidata.quantity import Quantity
from datetime import date


class HttpRequests:

    @staticmethod
    def get_entity_claims(entity_id: str) -> list:
        client = Client()
        entity: Entity = client.get(EntityId(entity_id), load=True)

        verbalized_claims = []

        for item in entity.lists():
            for value in item[1]:
                value_str: str = ''

                if type(value) is str:
                    value_str = value
                elif type(value) is Entity:
                    value_str = value.label["en"]
                elif type(value) is int:
                    value_str = str(value)
                elif type(value) is date:
                    value_str = value.strftime("%d/%m/%Y, %H:%M:%S")
                elif type(value) is Quantity:
                    value_str = str(Quantity)
                else:
                    print(f'{type(value)} not supported')

                if value_str != '':
                    verbalized_claims.append(f"{entity.label} {item[0].label} {value_str}")

        return verbalized_claims
