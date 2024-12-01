import json
import re
import gzip


def extract_entities_lines():
    mapping: dict[str, int] = {}

    with gzip.open('../../../data/wikidata-20240101-all.json.gz', 'rt') as fp:
        line_pos = 0

        for line in fp:
            if not line.startswith('{'):
                line_pos += len(line)
                continue

            first_match = re.search(r"\"id\":\"[QP]\d+\"", line)

            if first_match:
                entity_id: str = first_match.group()[6:-1]

                mapping[entity_id] = line_pos

            line_pos += len(line)

    with open('../../../data/entities_mapping.json', 'wt') as fp:
        json.dump(mapping, fp)


def main():
    extract_entities_lines()


if __name__ == '__main__':
    main()
