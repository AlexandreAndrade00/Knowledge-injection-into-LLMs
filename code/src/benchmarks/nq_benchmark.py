from benchmarks.benchmark import Benchmark
from plm.plm import PLM
from datasets import load_dataset
from typing import Optional, Iterable
from concurrent.futures import ThreadPoolExecutor
from functools import reduce


class NaturalQuestionsBenchmark(Benchmark):

    def __init__(self, model: PLM):
        super().__init__(model)
        self.dataset = load_dataset("natural_questions", split="train")

    @staticmethod
    def __extract_answer(data) -> Optional[str]:
        if len(data["annotations"]["short_answers"][0]["start_token"]) > 0:
            return " ".join(
                data["document"]["tokens"]["token"][data["annotations"]["short_answers"][0]["start_token"][0]:
                                                    data["annotations"]["short_answers"][0]["end_token"][0]])
        elif data["annotations"]["long_answer"][0]["start_token"] != -1:
            return " ".join(data["document"]["tokens"]["token"][data["annotations"]["long_answer"][0]["start_token"]:
                                                                data["annotations"]["long_answer"][0]["end_token"]])
        elif data["annotations"]["yes_no_answer"] != "NONE":
            return data["annotations"]["yes_no_answer"]
        else:
            return None

    def data(self) -> Iterable[dict[str, str]]:
        iterations = 10

        for data in self.dataset:
            if iterations == 0:
                break

            iterations = iterations - 1

            question = data["question"]["text"]

            context_list: list[str] = super().ranked_entities_statements(question)

            context: str = reduce(lambda s1, s2: f"{s1}\n{s2}\n", context_list) if len(context_list) > 0 else ""

            yield {"question": question, "context": context}
