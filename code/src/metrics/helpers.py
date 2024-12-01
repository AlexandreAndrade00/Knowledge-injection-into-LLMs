import json
from pathlib import Path
import seaborn as sns


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

    sns.displot(sim, kde=True)
    # sns.displot(rouge, kde=True)
