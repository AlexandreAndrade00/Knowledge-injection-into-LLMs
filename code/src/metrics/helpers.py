import json
from pathlib import Path
import seaborn as sns
import pandas as pd


def read_metrics(path: str) -> list[dict[str, float]]:
    file = Path(path)

    if not file.is_file():
        print(f"File {path} not found.")

        return []

    with open(path, "r") as fp:
        json_data = json.load(fp)

        return [elem["metrics"] for elem in json_data]


def plot_from_file(path: str) -> None:
    metrics: list[dict[str, float]] = read_metrics(path)[:100]

    sim: list[float] = [e["semantic-similarity"] for e in metrics if len(e) != 0]
    # rouge: list[float] = [e["rouge-L"] for e in metrics if len(e) != 0]

    sns.displot(sim, kde=True, rug=True)
    # sns.displot(rouge, kde=True)


def extract_metrics_from_multiple_files(
    base_path: str, k: list[int], hops: list[int], model_name: str
):
    data: list[list[Any]] = []

    for e_k in k:
        for hop in hops:
            metrics = read_metrics(
                base_path + f"/{model_name}-context-hop_{hop}-k_{e_k}.json"
            )[:100]

            for idx, metric in enumerate(metrics):
                data.append(
                    [idx, e_k, hop, metric["rouge-1"], metric["semantic-similarity"]]
                )

    dataframe = pd.DataFrame(
        data, columns=["q_index", "k", "hops", "rouge-1", "sem-sim"]
    )

    dataframe.to_csv(base_path + "/extracted_metrics.csv", index=False)


extract_metrics_from_multiple_files("../../outputs", [10], [1], "llama3b")
