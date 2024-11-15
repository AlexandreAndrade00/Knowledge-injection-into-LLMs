from abc import abstractmethod, ABC
from typing import Optional, Iterable
from plm.plm import PLM
from http_requests import HttpRequests
from knowledge_mixin import KnowledgeMixin
import json
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor


class Benchmark(ABC, KnowledgeMixin):
    model: PLM
    http_requests: HttpRequests = HttpRequests()
    output_path: Optional[str] = None

    def __init__(self, model: PLM):
        self.model = model

    def set_model(self, model: PLM) -> None:
        self.model = model

    def set_output_path(self, path: str) -> None:
        self.output_path = path

    @abstractmethod
    def data(self) -> Iterable[dict[str, str]]:
        raise NotImplementedError

    def run(self) -> None:
        if self.output_path is not None:
            Path(self.output_path).mkdir(parents=True, exist_ok=True)
            with open(self.output_path + "/results.json", "w"):
                pass

        pool = ThreadPoolExecutor(max_workers=10)

        for result in self.model.benchmark(self.data()):
            pool.submit(lambda: self.self.__write_to_file(result))

    def __write_to_file(self, result):
        print(result, flush=True)
        with open(self.output_path + "/results.json", "a") as fp:
            json.dump(result, fp)
