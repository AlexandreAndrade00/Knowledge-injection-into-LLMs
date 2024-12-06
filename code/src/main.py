from benchmarks.nq_benchmark import NaturalQuestionsBenchmark
from benchmarks.benchmark import Benchmark
from plm.llama_builder import LlamaBuilder
from plm.plm import PLM
from oracle import Oracle
import nltk
from torch.cuda import set_per_process_memory_fraction

from knowledge_mixin import DataOrigin


def main():
    # set_per_process_memory_fraction(0.8)

    nltk.download("punkt")
    nltk.download("punkt_tab")

    plm: PLM = LlamaBuilder().set_params(3).build()
    # oracle = Oracle(plm, False)
    #
    # oracle.set_question("What is the name of the last episode of The Walking Dead?")
    #
    # print(oracle.answer())

    nq_bmk = NaturalQuestionsBenchmark(
        plm,
        None,
        runs=100,
        use_context=True,
        data_origin=DataOrigin.LOCAL,
    )
    nq_bmk.set_output_path("/home/alexandre/dev/tsln_project/outputs")

    nq_bmk.set_hops(1)

    nq_bmk.set_k(10)
    nq_bmk.run()


if __name__ == "__main__":
    main()
