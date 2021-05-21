from typing import Dict, Iterable, Iterator, Set, Tuple
from pathlib import Path


class Lexicon:
    @classmethod
    def symmetrize(cls, direct_lexicon: "Lexicon", inverse_lexicon: "Lexicon") -> "Lexicon":
        src_words: Set[str] = set(direct_lexicon.source_words)
        src_words.update(inverse_lexicon.target_words)

        trg_words: Set[str] = set(inverse_lexicon.source_words)
        trg_words.update(direct_lexicon.target_words)

        lexicon = Lexicon()
        for src_word in src_words:
            for trg_word in trg_words:
                direct_prob = direct_lexicon[src_word, trg_word]
                inverse_prob = inverse_lexicon[trg_word, src_word]
                prob = max(direct_prob, inverse_prob)
                lexicon[src_word, trg_word] = prob
        return lexicon

    def __init__(self) -> None:
        self._table: Dict[str, Dict[str, float]] = {}

    def __getitem__(self, indices: Tuple[str, str]) -> float:
        src_word, trg_word = indices
        src_entry = self._table.get(src_word)
        if src_entry is None:
            return 0
        return src_entry.get(trg_word, 0)

    def __setitem__(self, indices: Tuple[str, str], value: float) -> None:
        if value == 0:
            return
        src_word, trg_word = indices
        src_entry = self._table.get(src_word)
        if src_entry is None:
            src_entry = {}
            self._table[src_word] = src_entry
        src_entry[trg_word] = value

    def __iter__(self) -> Iterator[Tuple[str, str, float]]:
        return (
            (src_word, trg_word, prob)
            for (src_word, trg_words) in self._table.items()
            for (trg_word, prob) in trg_words.items()
        )

    @property
    def source_words(self) -> Iterable[str]:
        return self._table.keys()

    @property
    def target_words(self) -> Iterable[str]:
        trg_words: Set[str] = set()
        for src_entry in self._table.values():
            trg_words.update(src_entry.keys())
        return trg_words

    def write(self, file_path: Path) -> None:
        with open(file_path, "w", encoding="utf-8", newline="\n") as file:
            for src_word, trg_word, prob in sorted(self, key=lambda t: (t[0], -t[2], t[1])):
                file.write(f"{src_word}\t{trg_word}\t{round(prob, 8)}\n")
