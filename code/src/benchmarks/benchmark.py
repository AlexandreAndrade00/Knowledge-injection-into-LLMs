from abc import abstractmethod, ABC
from typing import Optional, Iterable
from plm.plm import PLM
from knowledge_mixin import KnowledgeMixin
import json
from pathlib import Path
import nltk
from nltk.translate.bleu_score import sentence_bleu
from rouge_score import rouge_scorer
from sentence_transformers import SentenceTransformer

from knowledge_mixin import DataOrigin


class Benchmark(ABC, KnowledgeMixin):
    model: PLM
    runs: int
    use_context: bool
    output_path: str
    hops: int
    k: int
    context_path: Optional[str]
    __text_sim_model = SentenceTransformer("all-MiniLM-L6-v2")

    def __init__(
        self,
        model: PLM,
        runs: int,
        use_context: bool,
        hops: int,
        data_origin: DataOrigin,
        context_path: str,
        k: int,
    ):
        self.model = model
        self.runs = runs
        self.use_context = use_context
        self.hops = hops
        self.k = k
        self.context_path = context_path
        if context_path is None:
            super().__init__(data_origin)

    def set_model(self, model: PLM) -> None:
        self.model = model

    def set_output_path(self, path: str) -> None:
        self.output_path = path

    def set_runs(self, runs: int) -> None:
        self.runs = runs

    def set_hops(self, hops: int) -> None:
        self.hops = hops

    def set_k(self, k: int) -> None:
        self.k = k

    @abstractmethod
    def data(self) -> Iterable[dict[str, str]]:
        raise NotImplementedError

    @abstractmethod
    def save_data(self, file_name: str) -> None:
        raise NotImplementedError

    def run(self) -> None:
        if self.output_path is None:
            raise Exception("Output path not set")

        Path(self.output_path).mkdir(parents=True, exist_ok=True)
        with open(self.output_path + "/" + self.__output_file_name(), "w") as fp:
            fp.write("[")

        use_comma: bool = False

        for result in self.model.benchmark(self.data()):
            self.evaluate(result)
            self.__write_to_file(result, use_comma)
            use_comma = True

        with open(self.output_path + "/" + self.__output_file_name(), "a") as fp:
            fp.write("]")

    def evaluate(self, data: dict) -> None:
        data["metrics"] = {}

        if data["gold"] == [-1]:
            return

        reference_tokenized = nltk.word_tokenize(data["gold"])
        candidate_tokenized = nltk.word_tokenize(data["answer"])

        data["metrics"]["bleu"] = sentence_bleu(
            reference_tokenized, candidate_tokenized
        )

        scorer = rouge_scorer.RougeScorer(["rouge1", "rougeL"], use_stemmer=True)
        scores = scorer.score(data["gold"], data["answer"])
        data["metrics"]["rouge-1"] = scores["rouge1"].fmeasure
        data["metrics"]["rouge-L"] = scores["rougeL"].fmeasure

        embeddings1 = self.__text_sim_model.encode(data["gold"])
        embeddings2 = self.__text_sim_model.encode(data["answer"])

        data["metrics"]["semantic-similarity"] = self.__text_sim_model.similarity(
            embeddings1, embeddings2
        ).item()

    def __write_to_file(self, result: dict[str, str], use_comma: bool):
        # print(result, flush=True)
        with open(self.output_path + "/" + self.__output_file_name(), "a") as fp:
            if use_comma:
                fp.write(",")

            json.dump(result, fp)

    def __output_file_name(self):
        return f"{self.model.model_name}-{'context' if self.use_context else 'no_context'}-hop_{self.hops}-k_{self.k}.json"
