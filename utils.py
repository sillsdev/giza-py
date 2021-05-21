from pathlib import Path
from typing import Iterable, Iterator

def write_corpus(corpus_path: Path, sentences: Iterable[str]) -> None:
    with open(corpus_path, "w", encoding="utf-8", newline="\n") as file:
        for sentence in sentences:
            file.write(sentence + "\n")


def load_corpus(corpus_path: Path) -> Iterator[str]:
    with open(corpus_path, "r", encoding="utf-8-sig") as in_file:
        for line in in_file:
            line = line.strip()
            yield line