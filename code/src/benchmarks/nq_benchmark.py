import json

from benchmarks.benchmark import Benchmark
from plm.plm import PLM
from datasets import load_dataset
from typing import Optional, Iterable
from functools import reduce
from pathlib import Path as Dir
from knowledge_mixin import Path


class NaturalQuestionsBenchmark(Benchmark):
    def __init__(self, model: PLM, runs: int = -1, use_context: bool = True, hops: int = 1):
        super().__init__(model, runs, use_context, hops)
        self.dataset = load_dataset("natural_questions", split="train")

    @staticmethod
    def __extract_answer(data) -> str:
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
            return ""

    def data(self) -> Iterable[dict[str, str]]:
        for data in self.dataset:
            if self.runs == 0:
                break

            if self.runs > 0:
                super().set_runs(self.runs - 1)

            question = data["question"]["text"]

            data = {"question": question, "gold": self.__extract_answer(data)}

            if self.use_context:
                context_list: list[Path] = super().ranked_entities_statements(question, hops=self.hops)

                context: str = reduce(lambda s1, s2: f"{s1}\n{s2}",
                                      set(map(lambda e: e.verbalized, context_list))) \
                    if len(context_list) > 0 else ""

                data["context"] = context

            yield data

    def save_data(self) -> None:
        if self.output_path is None:
            raise Exception("Output path not set")

        Dir(self.output_path).mkdir(parents=True, exist_ok=True)

        use_comma: bool = False

        with open(self.output_path + "/data.json", "a") as fp:
            fp.write('[')

            json.dump({"context": self.use_context, "hops": self.hops}, fp)

            fp.write(',')

            for result in self.data():
                if use_comma:
                    fp.write(',')

                json.dump(result, fp)

                use_comma = True

            fp.write(']')
