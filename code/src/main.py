from benchmarks.nq_benchmark import NaturalQuestionsBenchmark
from benchmarks.benchmark import Benchmark
from plm.llama_builder import LlamaBuilder
from plm.plm import PLM
from oracle import Oracle


def main():
    plm: PLM = LlamaBuilder().build()
    oracle = Oracle(plm)

    oracle.set_question("Which is the capital of France?")

    print(oracle.answer())

    #nq_bmk = NaturalQuestionsBenchmark(plm)
    #nq_bmk.set_output_path("/home/alexandre/dev/tsln_project/outputs")
    #nq_bmk.run()


if __name__ == "__main__":
    main()
