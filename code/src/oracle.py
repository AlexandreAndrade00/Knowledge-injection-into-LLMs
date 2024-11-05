from refined.inference.processor import Refined
from plm.plm import PLM
from http_requests import HttpRequests


class Oracle:
    __model: PLM
    __refined: Refined
    __http_requests: HttpRequests = HttpRequests()

    def __init__(self, model: PLM):
        self.__model = model
        self.__refined = Refined.from_pretrained(model_name='wikipedia_model',
                                                 entity_set='wikipedia',
                                                 use_precomputed_descriptions=True)

    def answer(self, model_input: str) -> str:
        spans = self.__refined.process_text(model_input)

        verbalized_rdf = [self.__http_requests.get_entity_claims(span.predicted_entity.wikidata_entity_id) for span in
                          spans]

        print(verbalized_rdf)

        # return self.__model.inference(model_input)
