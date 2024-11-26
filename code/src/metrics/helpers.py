import json
from pathlib import Path


def read_metrics(path: str) -> list[dict[str, float]]:
    file = Path(path)

    if not file.is_file():
        print(f"File {path} not found.")

        return []

    with open(path, "r") as fp:
        json_data = json.load(fp)

        return [elem["metrics"] for elem in json_data]
