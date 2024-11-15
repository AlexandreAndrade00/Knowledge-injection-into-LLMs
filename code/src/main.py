from benchmarks.nq_benchmark import NaturalQuestionsBenchmark
from benchmarks.benchmark import Benchmark
from plm.llama_builder import LlamaBuilder
from plm.plm import PLM


def main():
    plm: PLM = LlamaBuilder().build()

    nq_bmk = NaturalQuestionsBenchmark(plm)
    nq_bmk.set_output_path("/home/alexandre/dev/tsln_project/outputs")
    nq_bmk.run()


if __name__ == "__main__":
    main()
