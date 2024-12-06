import json

from benchmarks.benchmark import Benchmark
from plm.plm import PLM
from datasets import load_dataset
from typing import Optional, Iterable, Any
from functools import reduce
from pathlib import Path as Dir
from knowledge_mixin import Path

from knowledge_mixin import DataOrigin


class NaturalQuestionsBenchmark(Benchmark):
    def __init__(
        self,
        model: PLM,
        context_path: Optional[str],
        runs: int,
        use_context: bool,
        hops: int = None,
        k: int = None,
        data_origin: DataOrigin = DataOrigin.SPARQL,
    ):
        super().__init__(model, runs, use_context, hops, data_origin, context_path, k)
        self.dataset = load_dataset("natural_questions", split="train")

    @staticmethod
    def __extract_answer(data) -> str:
        if len(data["annotations"]["short_answers"][0]["start_token"]) > 0:
            return " ".join(
                data["document"]["tokens"]["token"][
                    data["annotations"]["short_answers"][0]["start_token"][0] : data[
                        "annotations"
                    ]["short_answers"][0]["end_token"][0]
                ]
            )
        elif data["annotations"]["long_answer"][0]["start_token"] != -1:
            return " ".join(
                data["document"]["tokens"]["token"][
                    data["annotations"]["long_answer"][0]["start_token"] : data[
                        "annotations"
                    ]["long_answer"][0]["end_token"]
                ]
            )
        elif data["annotations"]["yes_no_answer"] != "NONE":
            return data["annotations"]["yes_no_answer"]
        else:
            return ""

    def data(self) -> Iterable[dict[str, str]]:
        for data in self.dataset:
            if self.runs == 0:
                break

            question = data["question"]["text"]

            data = {
                "id": data["id"],
                "question": question,
                "gold": self.__extract_answer(data),
            }

            if data["gold"] == [-1]:
                continue

            if self.runs > 0:
                super().set_runs(self.runs - 1)

            if self.use_context:
                if self.context_path is not None:
                    with open(self.context_path, "rt") as fp:
                        all_context: dict[str, Any] = json.load(fp)

                        if all_context["config"]["hops"] != self.hops:
                            raise Exception(
                                "Context file have a different number of hops"
                            )

                        context: str = all_context[data["id"]]["context"]
                else:
                    context_list: list[Path] = super().ranked_entities_statements(
                        question, hops=self.hops, k=self.k
                    )

                    context: str = (
                        reduce(
                            lambda s1, s2: f"{s1}\n{s2}",
                            set(map(lambda e: e.verbalized, context_list)),
                        )
                        if len(context_list) > 0
                        else ""
                    )

                data["context"] = context

            yield data

    def save_data(self) -> None:
        if self.output_path is None:
            raise Exception("Output path not set")

        Dir(self.output_path).mkdir(parents=True, exist_ok=True)

        data: dict[str, Any] = {"config": {"hops": self.hops}}

        for result in self.data():
            data[result["id"]] = {"id": result["id"], "context": result["context"]}

        with open(self.output_path + "/data.json", "w") as fp:
            json.dump(data, fp)
