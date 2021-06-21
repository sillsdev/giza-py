import codecs
import os
from pathlib import Path
from typing import IO, Iterable, Iterator, Set, Tuple


def write_corpus(corpus_path: Path, sentences: Iterable[str]) -> None:
    with open(corpus_path, "w", encoding="utf-8", newline="\n") as file:
        for sentence in sentences:
            file.write(sentence + "\n")


def load_corpus(corpus_path: Path) -> Iterator[str]:
    with open(corpus_path, "r", encoding="utf-8-sig") as in_file:
        for line in in_file:
            line = line.strip()
            yield line


def parse_giza_alignments(alignments_file: IO[str]) -> Iterable[Set[Tuple[int, int]]]:
    line_index = 0
    for line in alignments_file:
        line = line.strip()
        if line.startswith("#"):
            line_index = 0
        elif line_index == 2:
            start = line.find("({")
            end = line.find("})")
            src_index = -1
            pairs: Set[Tuple[int, int]] = set()
            while start != -1 and end != -1:
                if src_index > -1:
                    trg_indices_str = line[start + 2 : end].strip()
                    trg_indices = trg_indices_str.split()
                    pairs.update(((src_index, int(trg_index) - 1) for trg_index in trg_indices))
                start = line.find("({", start + 2)
                if start >= 0:
                    end = line.find("})", end + 2)
                    src_index += 1
            yield pairs
        line_index += 1


def remove_bom_inplace(path):
    """Removes BOM mark, if it exists, from a file and rewrites it in-place"""
    buffer_size = 4096
    bom_length = len(codecs.BOM_UTF8)

    with open(path, "r+b") as fp:
        chunk = fp.read(buffer_size)
        if chunk.startswith(codecs.BOM_UTF8):
            i = 0
            chunk = chunk[bom_length:]
            while chunk:
                fp.seek(i)
                fp.write(chunk)
                i += len(chunk)
                fp.seek(bom_length, os.SEEK_CUR)
                chunk = fp.read(buffer_size)
            fp.seek(-bom_length, os.SEEK_CUR)
            fp.truncate()
