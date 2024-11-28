from benchmarks.nq_benchmark import NaturalQuestionsBenchmark
from benchmarks.benchmark import Benchmark
from plm.llama_builder import LlamaBuilder
from plm.plm import PLM
from oracle import Oracle
import nltk


def main():
    nltk.download('punkt')
    nltk.download('punkt_tab')

    plm: PLM = LlamaBuilder().build()
    # oracle = Oracle(plm, False)
    #
    # oracle.set_question("What is the name of the last episode of The Walking Dead?")
    #
    # print(oracle.answer())

    nq_bmk = NaturalQuestionsBenchmark(plm, 100, use_context=True, hops=2)
    nq_bmk.set_output_path("/home/alexandre/dev/tsln_project/outputs")
    nq_bmk.run()


if __name__ == "__main__":
    main()
