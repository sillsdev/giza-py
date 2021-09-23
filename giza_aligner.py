import platform
import shutil
import subprocess
from bisect import insort_left
from math import ceil
from pathlib import Path
from typing import IO, Any, Dict, Iterator, List, Optional, Set, Tuple

from lexicon import Lexicon
from utils import load_corpus, parse_giza_alignments, remove_bom_inplace, write_corpus

MAX_SENT_LENGTH = 101
PROB_SMOOTH = 1e-7
IBM4_SMOOTH_FACTOR = 0.2


class GizaAligner:
    def __init__(
        self,
        bin_dir: Path,
        model_dir: Path,
        m1: Optional[int] = None,
        m2: Optional[int] = None,
        mh: Optional[int] = None,
        m3: Optional[int] = None,
        m4: Optional[int] = None,
    ) -> None:
        self.bin_dir = bin_dir
        self.model_dir = model_dir
        self.m1 = m1
        self.m2 = m2
        self.mh = mh
        self.m3 = m3
        self.m4 = m4

    @property
    def file_suffix(self) -> str:
        suffix = ""
        if self.m3 is None or self.m3 > 0 or self.m4 is None or self.m4 > 0:
            suffix = "3.final"
        elif self.mh is None or self.mh > 0:
            suffix = f"hmm.{5 if self.mh is None else self.mh}"
        elif self.m2 is not None and self.m2 > 0:
            suffix = f"2.{self.m2}"
        elif self.m1 is None or self.m1 > 0:
            suffix = f"1.{5 if self.m1 is None else self.m1}"
        return suffix

    def train(self, src_file_path: Path, trg_file_path: Path, quiet: bool = False) -> None:
        self.model_dir.mkdir(exist_ok=True)
        dest_src_file_path = self.model_dir / "src.txt"
        shutil.copyfile(src_file_path, dest_src_file_path)
        src_file_path = dest_src_file_path
        dest_trg_file_path = self.model_dir / "trg.txt"
        shutil.copyfile(trg_file_path, dest_trg_file_path)
        trg_file_path = dest_trg_file_path

        remove_bom_inplace(src_file_path)
        remove_bom_inplace(trg_file_path)

        if self.m4 is None or self.m4 > 0:
            self._execute_mkcls(src_file_path, "src", quiet)
            self._execute_mkcls(trg_file_path, "trg", quiet)

        src_trg_snt_file_path, trg_src_snt_file_path = self._execute_plain2snt(
            src_file_path, trg_file_path, "src", "trg", quiet
        )

        self._execute_snt2cooc(src_trg_snt_file_path, quiet)
        self._execute_snt2cooc(trg_src_snt_file_path, quiet)

        src_trg_prefix = src_trg_snt_file_path.with_suffix("")
        src_trg_output_prefix = src_trg_prefix.parent / (src_trg_prefix.name + "_invswm")
        self._execute_mgiza(src_trg_snt_file_path, src_trg_output_prefix, quiet)
        src_trg_alignments_file_path = src_trg_output_prefix.with_suffix(f".A{self.file_suffix}.all")
        self._save_alignments(src_trg_output_prefix, src_trg_alignments_file_path)

        trg_src_output_prefix = src_trg_prefix.parent / (src_trg_prefix.name + "_swm")
        self._execute_mgiza(trg_src_snt_file_path, trg_src_output_prefix, quiet)
        trg_src_alignments_file_path = trg_src_output_prefix.with_suffix(f".A{self.file_suffix}.all")
        self._save_alignments(trg_src_output_prefix, trg_src_alignments_file_path)

    def align(
        self,
        alignments_file_path: Path,
        include_probs: bool = False,
        sym_heuristic: str = "grow-diag-final-and",
    ) -> None:
        src_trg_alignments_file_path = self.model_dir / f"src_trg_invswm.A{self.file_suffix}.all"
        trg_src_alignments_file_path = self.model_dir / f"src_trg_swm.A{self.file_suffix}.all"
        sym_alignments_file_path = self.model_dir / "alignments.txt"
        self._symmetrize(
            src_trg_alignments_file_path,
            trg_src_alignments_file_path,
            sym_alignments_file_path,
            sym_heuristic,
        )

        src_file_path = self.model_dir / "src.txt"
        trg_file_path = self.model_dir / "trg.txt"

        with open(alignments_file_path, "w", encoding="utf-8", newline="\n") as alignments_file, open(
            sym_alignments_file_path,
            "r",
            encoding="utf-8-sig",
        ) as sym_alignments_file:
            alignment_probs_data: Any = None
            direct_alignments_file: Optional[IO] = None
            inverse_alignments_file: Optional[IO] = None
            direct_alignments: Optional[Iterator[Set[Tuple[int, int]]]] = None
            inverse_alignments: Optional[Iterator[Set[Tuple[int, int]]]] = None
            if include_probs:
                alignment_probs_data = self._init_alignment_probs_data()
                direct_alignments_file = open(src_trg_alignments_file_path, "r", encoding="utf-8-sig")
                direct_alignments = iter(parse_giza_alignments(direct_alignments_file))
                inverse_alignments_file = open(trg_src_alignments_file_path, "r", encoding="utf-8-sig")
                inverse_alignments = iter(parse_giza_alignments(inverse_alignments_file))
            try:
                for src_str, trg_str in zip(load_corpus(src_file_path), load_corpus(trg_file_path)):
                    if len(src_str) == 0 or len(trg_str) == 0:
                        alignments_file.write("\n")
                        continue

                    src_tokens = src_str.split()
                    trg_tokens = trg_str.split()
                    alignment_str = sym_alignments_file.readline().strip()

                    if direct_alignments is not None and inverse_alignments is not None:
                        direct_alignment = next(direct_alignments)
                        inverse_alignment = next(inverse_alignments)

                        direct_probs = self._get_alignment_probs(
                            alignment_probs_data, src_tokens, trg_tokens, direct_alignment, True
                        )
                        inverse_probs = self._get_alignment_probs(
                            alignment_probs_data, trg_tokens, src_tokens, inverse_alignment, False
                        )

                        new_alignment_str = ""
                        for word_pair_str in alignment_str.split():
                            src_index_str, trg_index_str = word_pair_str.split("-", maxsplit=2)
                            src_index = int(src_index_str)
                            trg_index = int(trg_index_str)
                            direct_prob = direct_probs.get((src_index, trg_index), 0.0)
                            inverse_prob = inverse_probs.get((trg_index, src_index), 0.0)
                            prob = round(max(direct_prob, inverse_prob), 8)
                            if len(new_alignment_str) != 0:
                                new_alignment_str += " "
                            new_alignment_str += f"{src_index}-{trg_index}:{prob}"
                        alignment_str = new_alignment_str
                    alignments_file.write(alignment_str + "\n")
            finally:
                if direct_alignments_file is not None:
                    direct_alignments_file.close()
                if inverse_alignments_file is not None:
                    inverse_alignments_file.close()

    def extract_lexicon(self, out_file_path: Path, threshold: float = 0.0) -> None:
        src_vocab = self._load_vocab("src")
        trg_vocab = self._load_vocab("trg")
        direct_lexicon = self._load_lexicon(src_vocab, trg_vocab, "invswm", threshold=threshold)
        inverse_lexicon = self._load_lexicon(trg_vocab, src_vocab, "swm", threshold=threshold)
        lexicon = Lexicon.symmetrize(direct_lexicon, inverse_lexicon, threshold=threshold)
        lexicon.write(out_file_path)

    def _execute_mkcls(self, input_file_path: Path, output_prefix: str, quiet: bool) -> None:
        mkcls_path = self.bin_dir / "mkcls"
        if platform.system() == "Windows":
            mkcls_path = mkcls_path.with_suffix(".exe")
        if not mkcls_path.is_file():
            raise RuntimeError("mkcls is not installed.")

        output_file_path = self.model_dir / f"{output_prefix}.vcb.classes"

        args: List[str] = [
            str(mkcls_path),
            "-n10",
            f"-p{input_file_path}",
            f"-V{output_file_path}",
        ]
        subprocess.run(args, stdout=subprocess.DEVNULL if quiet else None, stderr=subprocess.DEVNULL if quiet else None)

    def _execute_plain2snt(
        self, src_file_path: Path, trg_file_path: Path, output_src_prefix: str, output_trg_prefix: str, quiet: bool
    ) -> Tuple[Path, Path]:
        plain2snt_path = self.bin_dir / "plain2snt"
        if platform.system() == "Windows":
            plain2snt_path = plain2snt_path.with_suffix(".exe")
        if not plain2snt_path.is_file():
            raise RuntimeError("plain2snt is not installed.")

        src_trg_snt_file_path = self.model_dir / f"{output_src_prefix}_{output_trg_prefix}.snt"
        trg_src_snt_file_path = self.model_dir / f"{output_trg_prefix}_{output_src_prefix}.snt"

        args: List[str] = [
            str(plain2snt_path),
            str(src_file_path),
            str(trg_file_path),
            "-vcb1",
            str(self.model_dir / f"{output_src_prefix}.vcb"),
            "-vcb2",
            str(self.model_dir / f"{output_trg_prefix}.vcb"),
            "-snt1",
            str(src_trg_snt_file_path),
            "-snt2",
            str(trg_src_snt_file_path),
        ]
        subprocess.run(args, stdout=subprocess.DEVNULL if quiet else None, stderr=subprocess.DEVNULL)
        return src_trg_snt_file_path, trg_src_snt_file_path

    def _execute_snt2cooc(self, snt_file_path: Path, quiet: bool) -> None:
        snt2cooc_path = self.bin_dir / "snt2cooc"
        if platform.system() == "Windows":
            snt2cooc_path = snt2cooc_path.with_suffix(".exe")
        if not snt2cooc_path.is_file():
            raise RuntimeError("snt2cooc is not installed.")

        snt_dir = snt_file_path.parent
        prefix = snt_file_path.stem
        prefix1, prefix2 = prefix.split("_", maxsplit=2)

        args: List[str] = [
            str(snt2cooc_path),
            str(self.model_dir / f"{prefix}.cooc"),
            str(snt_dir / f"{prefix1}.vcb"),
            str(snt_dir / f"{prefix2}.vcb"),
            str(snt_file_path),
        ]
        subprocess.run(args, stdout=subprocess.DEVNULL if quiet else None, stderr=subprocess.DEVNULL)

    def _execute_mgiza(self, snt_file_path: Path, output_path: Path, quiet: bool) -> None:
        mgiza_path = self.bin_dir / "mgiza"
        if platform.system() == "Windows":
            mgiza_path = mgiza_path.with_suffix(".exe")
        if not mgiza_path.is_file():
            raise RuntimeError("mgiza is not installed.")

        snt_dir = snt_file_path.parent
        prefix = snt_file_path.stem
        prefix1, prefix2 = prefix.split("_", maxsplit=2)

        args: List[str] = [
            str(mgiza_path),
            "-C",
            str(snt_file_path),
            "-CoocurrenceFile",
            str(snt_dir / f"{prefix}.cooc"),
            "-S",
            str(snt_dir / f"{prefix1}.vcb"),
            "-T",
            str(snt_dir / f"{prefix2}.vcb"),
            "-o",
            str(output_path),
        ]
        if self.m1 is not None:
            args.extend(["-m1", str(self.m1)])
        if self.m2 is not None and (self.mh is None or self.mh == 0):
            args.extend(["-m2", str(self.m2)])
            if self.mh is None:
                args.extend(["-mh", "0"])
        if self.mh is not None:
            args.extend(["-mh", str(self.mh)])
        if self.m3 is not None:
            args.extend(["-m3", str(self.m3)])
        if self.m4 is not None:
            args.extend(["-m4", str(self.m4)])

        if self.m3 == 0 and self.m4 == 0:
            if self.mh is None or self.mh > 0:
                args.extend(["-th", str(5 if self.mh is None else self.mh)])
            elif self.m2 is not None and self.m2 > 0:
                args.extend(["-t2", str(self.m2)])
            elif self.m1 is None or self.m1 > 0:
                args.extend(["-t1", str(5 if self.m1 is None else self.m1)])
        subprocess.run(args, stdout=subprocess.DEVNULL if quiet else None, stderr=subprocess.DEVNULL if quiet else None)

    def _save_alignments(self, model_prefix: Path, output_file_path: Path) -> None:
        alignments: List[Tuple[int, str]] = []
        for input_file_path in model_prefix.parent.glob(model_prefix.name + f".A{self.file_suffix}.part*"):
            with open(input_file_path, "r", encoding="utf-8") as in_file:
                line_index = 0
                segment_index = 0
                cur_alignment: str = ""
                for line in in_file:
                    cur_alignment += line
                    alignment_line_index = line_index % 3
                    if alignment_line_index == 0:
                        start = line.index("(")
                        end = line.index(")")
                        segment_index = int(line[start + 1 : end])
                    elif alignment_line_index == 2:
                        alignments.append((segment_index, cur_alignment.strip()))
                        cur_alignment = ""
                    line_index += 1

        write_corpus(
            output_file_path,
            map(lambda a: str(a[1]), sorted(alignments, key=lambda a: a[0])),
        )

    def _symmetrize(
        self,
        direct_align_path: Path,
        inverse_align_path: Path,
        output_path: Path,
        sym_heuristic: str,
    ) -> None:
        args: List[str] = [
            "dotnet",
            "machine",
            "symmetrize",
            str(direct_align_path),
            str(inverse_align_path),
            str(output_path),
            "-sh",
            sym_heuristic,
        ]
        subprocess.run(args, stdout=subprocess.DEVNULL)

    def _init_alignment_probs_data(self) -> Any:
        return None

    def _get_alignment_probs(
        self, data: Any, src_words: List[str], trg_words: List[str], alignment: Set[Tuple[int, int]], is_direct: bool
    ) -> Dict[Tuple[int, int], float]:
        return {word_pair: 1.0 / (len(src_words) + 1) for word_pair in alignment}

    def _load_vocab(self, side: str) -> List[str]:
        vocab_path = self.model_dir / f"{side}.vcb"
        vocab: List[str] = ["NULL", "UNK"]
        for line in load_corpus(vocab_path):
            index_str, word, _ = line.split()
            assert int(index_str) == len(vocab)
            vocab.append(word)
        return vocab

    def _load_lexicon(
        self,
        src_vocab: List[str],
        trg_vocab: List[str],
        align_model: str,
        threshold: float = 0.0,
        include_special_tokens: bool = False,
    ) -> Lexicon:
        lexicon = Lexicon()
        model_path = self.model_dir / f"src_trg_{align_model}.t{self.file_suffix}"
        for line in load_corpus(model_path):
            src_index_str, trg_index_str, prob_str = line.split(maxsplit=3)
            src_index = int(src_index_str)
            trg_index = int(trg_index_str)
            if include_special_tokens or (src_index > 1 and trg_index > 1):
                src_word = src_vocab[src_index]
                trg_word = trg_vocab[trg_index]
                prob = float(prob_str)
                if prob > threshold:
                    lexicon[src_word, trg_word] = prob
        return lexicon


class Ibm1GizaAligner(GizaAligner):
    def __init__(self, bin_dir: Path, model_dir: Path, m1: Optional[int] = None) -> None:
        super().__init__(bin_dir, model_dir, m1=m1, mh=0, m3=0, m4=0)


class Ibm2GizaAligner(GizaAligner):
    def __init__(self, bin_dir: Path, model_dir: Path, m1: Optional[int] = None, m2: Optional[int] = None) -> None:
        super().__init__(bin_dir, model_dir, m1=m1, m2=5 if m2 is None else m2, mh=0, m3=0, m4=0)

    def _init_alignment_probs_data(self) -> Any:
        return {
            "direct_alignment_table": self._load_alignment_table("invswm"),
            "inverse_alignment_table": self._load_alignment_table("swm"),
        }

    def _save_alignments(self, model_prefix: Path, output_file_path: Path) -> None:
        shutil.move(str(model_prefix) + f".A{self.file_suffix}", output_file_path)

    def _get_alignment_probs(
        self, data: Any, src_words: List[str], trg_words: List[str], alignment: Set[Tuple[int, int]], is_direct: bool
    ) -> Dict[Tuple[int, int], float]:
        alignment_table: Dict[Tuple[int, int], Dict[int, float]]
        if is_direct:
            alignment_table = data["direct_alignment_table"]
        else:
            alignment_table = data["inverse_alignment_table"]

        probs: Dict[Tuple[int, int], float] = {}
        for src_index, trg_index in alignment:
            i = src_index + 1
            j = trg_index + 1
            prob = 0.0
            elem = alignment_table.get((j, len(src_words)))
            if elem is not None:
                prob = elem.get(i, 0.0)
            probs[(src_index, trg_index)] = max(PROB_SMOOTH, prob)
        return probs

    def _load_alignment_table(self, align_model: str) -> Dict[Tuple[int, int], Dict[int, float]]:
        table: Dict[Tuple[int, int], Dict[int, float]] = {}
        totals: Dict[Tuple[int, int], float] = {}
        ext = "ap" if platform.system() == "Windows" else "a"
        for line in load_corpus(self.model_dir / f"src_trg_{align_model}.{ext}{self.file_suffix}"):
            fields = line.split(maxsplit=5)
            i = int(fields[0])
            j = int(fields[1])
            slen = int(fields[2])
            count = float(fields[4])
            key = (j, slen)
            counts = table.get(key)
            if counts is None:
                counts = {}
                table[key] = counts
            counts[i] = count
            total = totals.get(key, 0.0)
            totals[key] = total + count

        for key, counts in table.items():
            total = totals[key]
            for j, count in counts.items():
                counts[j] = count / total

        return table


def normalize(values: List[float]) -> None:
    sum_values = sum(values)
    for i in range(len(values)):
        if sum_values > 0:
            values[i] /= sum_values
        else:
            values[i] = 1.0 / len(values)


def smooth(values: List[float], p: float) -> None:
    pp = p / len(values)
    for i in range(len(values)):
        values[i] = (1.0 - p) * values[i] + pp


class HmmGizaAligner(GizaAligner):
    def __init__(self, bin_dir: Path, model_dir: Path, m1: Optional[int] = None, mh: Optional[int] = None) -> None:
        super().__init__(bin_dir, model_dir, m1=m1, mh=mh, m3=0, m4=0)

    def _init_alignment_probs_data(self) -> Any:
        return {
            "direct_alignment_table": self._load_alignment_table("invswm"),
            "direct_alpha_table": self._load_alpha_table("invswm"),
            "inverse_alignment_table": self._load_alignment_table("swm"),
            "inverse_alpha_table": self._load_alpha_table("swm"),
        }

    def _get_alignment_probs(
        self, data: Any, src_words: List[str], trg_words: List[str], alignment: Set[Tuple[int, int]], is_direct: bool
    ) -> Dict[Tuple[int, int], float]:
        alignment_table: Dict[int, float]
        alpha_table: Dict[int, List[float]]
        if is_direct:
            alignment_table = data["direct_alignment_table"]
            alpha_table = data["direct_alpha_table"]
        else:
            alignment_table = data["inverse_alignment_table"]
            alpha_table = data["inverse_alpha_table"]

        probs_table: List[List[float]] = []
        for i1 in range(len(src_words) * 2):
            i1_real = i1 % len(src_words)
            al: List[float] = []
            for i2 in range(len(src_words)):
                al.append(alignment_table.get(i1_real - i2, 1.0 / (2 * (len(src_words) - 1))))
            normalize(al)
            smooth(al, 0.2)
            i1_probs: List[float] = []
            for i2 in range(len(src_words) * 2):
                i2_real = i2 % len(src_words)
                empty_i2 = i2 >= len(src_words)
                if empty_i2:
                    prob = 0.4 if i1_real == i2_real else 0
                else:
                    prob = al[i2_real]
                i1_probs.append(prob)
            normalize(i1_probs)
            probs_table.append(i1_probs)

        alpha = alpha_table[len(src_words)]

        asymm_al: List[int] = [-1] * len(trg_words)
        for src_index, trg_index in alignment:
            asymm_al[trg_index] = src_index

        probs: Dict[Tuple[int, int], float] = {}
        prev_src_index = -1
        for trg_index in range(len(asymm_al)):
            src_index = asymm_al[trg_index]
            if prev_src_index == -1:
                if src_index == -1:
                    src_index = len(src_words)
                else:
                    probs[(src_index, trg_index)] = alpha[src_index]
            elif src_index == -1:
                src_index = prev_src_index + len(src_words)
            else:
                probs[(src_index, trg_index)] = probs_table[prev_src_index][src_index]
            prev_src_index = src_index
        return probs

    def _load_alignment_table(self, align_model: str) -> Dict[int, float]:
        table: Dict[int, float] = {}
        for line in load_corpus(self.model_dir / f"src_trg_{align_model}.h{self.file_suffix}"):
            fields = line.split()
            for i in range(7, len(fields), 2):
                pos = int(fields[i])
                value = float(fields[i + 1])
                table[pos] = value

        return table

    def _load_alpha_table(self, align_model: str) -> Dict[int, List[float]]:
        table: Dict[int, List[float]] = {}
        for line in load_corpus(self.model_dir / f"src_trg_{align_model}.h{self.file_suffix}.alpha"):
            fields = line.split()
            src_len = int(fields[0]) / 2
            values: List[float] = []
            for i in range(2, len(fields)):
                value = float(fields[i])
                values.append(value)
            normalize(values)
            table[src_len] = values

        return table


class Ibm3GizaAligner(GizaAligner):
    def __init__(
        self,
        bin_dir: Path,
        model_dir: Path,
        m1: Optional[int] = None,
        m2: Optional[int] = None,
        mh: Optional[int] = None,
        m3: Optional[int] = None,
    ) -> None:
        super().__init__(bin_dir, model_dir, m1=m1, m2=m2, mh=mh, m3=m3, m4=0)

    def _init_alignment_probs_data(self) -> Any:
        return {
            "direct_distortion_table": self._load_distortion_table("invswm"),
            "inverse_distortion_table": self._load_distortion_table("swm"),
        }

    def _get_alignment_probs(
        self, data: Any, src_words: List[str], trg_words: List[str], alignment: Set[Tuple[int, int]], is_direct: bool
    ) -> Dict[Tuple[int, int], float]:
        distortion_table: Dict[Tuple[int, int], Dict[int, float]]
        if is_direct:
            distortion_table = data["direct_distortion_table"]
        else:
            distortion_table = data["inverse_distortion_table"]

        probs: Dict[Tuple[int, int], float] = {}
        for src_index, trg_index in alignment:
            i = src_index + 1
            j = trg_index + 1
            prob = 0.0
            elem = distortion_table.get((i, len(trg_words)))
            if elem is not None:
                prob = elem.get(j, 0.0)
            probs[(src_index, trg_index)] = max(PROB_SMOOTH, prob)
        return probs

    def _load_distortion_table(self, align_model: str) -> Dict[Tuple[int, int], Dict[int, float]]:
        table: Dict[Tuple[int, int], Dict[int, float]] = {}
        for line in load_corpus(self.model_dir / f"src_trg_{align_model}.d{self.file_suffix}"):
            fields = line.split(maxsplit=5)
            j = int(fields[0])
            i = int(fields[1])
            tlen = int(fields[3])
            prob = float(fields[4])
            key = (i, tlen)
            probs = table.get(key)
            if probs is None:
                probs = {}
                table[key] = probs
            probs[j] = prob
        return table


class Ibm4GizaAligner(GizaAligner):
    def __init__(
        self,
        bin_dir: Path,
        model_dir: Path,
        m1: Optional[int] = None,
        m2: Optional[int] = None,
        mh: Optional[int] = None,
        m3: Optional[int] = None,
        m4: Optional[int] = None,
    ) -> None:
        super().__init__(bin_dir, model_dir, m1=m1, m2=m2, mh=mh, m3=m3, m4=m4)

    def _init_alignment_probs_data(self) -> Any:
        return {
            "src_word_classes": self._load_word_classes("src"),
            "trg_word_classes": self._load_word_classes("trg"),
            "direct_head_distortion_table": self._load_head_distortion_table("invswm"),
            "inverse_head_distortion_table": self._load_head_distortion_table("swm"),
            "direct_nonhead_distortion_table": self._load_nonhead_distortion_table("invswm"),
            "inverse_nonhead_distortion_table": self._load_nonhead_distortion_table("swm"),
        }

    def _get_alignment_probs(
        self, data: Any, src_words: List[str], trg_words: List[str], alignment: Set[Tuple[int, int]], is_direct: bool
    ) -> Dict[Tuple[int, int], float]:
        head_distortion_table: Dict[Tuple[int, int], Dict[int, float]]
        nonhead_distortion_table: Dict[int, Dict[int, float]]
        src_classes: Dict[str, int]
        trg_classes: Dict[str, int]
        if is_direct:
            head_distortion_table = data["direct_head_distortion_table"]
            nonhead_distortion_table = data["direct_nonhead_distortion_table"]
            src_classes = data["src_word_classes"]
            trg_classes = data["trg_word_classes"]
        else:
            head_distortion_table = data["inverse_head_distortion_table"]
            nonhead_distortion_table = data["inverse_nonhead_distortion_table"]
            src_classes = data["trg_word_classes"]
            trg_classes = data["src_word_classes"]

        cepts: List[List[int]] = [[] for _ in range(0, len(src_words) + 1)]
        for src_index, trg_index in alignment:
            i = src_index + 1
            j = trg_index + 1
            insort_left(cepts[i], j)

        probs: Dict[Tuple[int, int], float] = {}
        for src_index, trg_index in alignment:
            i = src_index + 1
            j = trg_index + 1
            t = trg_words[j - 1]
            trg_word_class = trg_classes[t]
            if cepts[i][0] == j:
                prev_cept = i - 1
                while prev_cept > 0 and len(cepts[prev_cept]) == 0:
                    prev_cept -= 1
                if prev_cept == 0:
                    src_word_class = 0
                    center = 0
                else:
                    s_prev_cept = src_words[prev_cept - 1]
                    src_word_class = src_classes[s_prev_cept]
                    center = int(ceil(sum(cepts[prev_cept]) / len(cepts[prev_cept])))
                dj = j - center
                prob = 0.0
                elem = head_distortion_table.get((src_word_class, trg_word_class))
                if elem is not None:
                    prob = elem.get(dj, 0.0)
                probs[(src_index, trg_index)] = max(
                    PROB_SMOOTH,
                    IBM4_SMOOTH_FACTOR / (2 * len(trg_words) - 1) + (1 - IBM4_SMOOTH_FACTOR) * prob,
                )
            else:
                pos_in_cept = cepts[i].index(j)
                prev_in_cept = cepts[i][pos_in_cept - 1]
                dj = j - prev_in_cept
                prob = 0.0
                elem = nonhead_distortion_table.get(trg_word_class)
                if elem is not None:
                    prob = elem.get(dj, 0.0)
                probs[(src_index, trg_index)] = max(
                    PROB_SMOOTH, IBM4_SMOOTH_FACTOR / (len(trg_words) - 1) + (1 - IBM4_SMOOTH_FACTOR) * prob
                )
        return probs

    def _load_word_classes(self, side: str) -> Dict[str, int]:
        word_classes: Dict[str, int] = {}
        classes: Dict[str, int] = {}
        for line in load_corpus(self.model_dir / f"{side}.vcb.classes"):
            word, word_class_str = line.split("\t", maxsplit=2)
            class_index = classes.get(word_class_str)
            if class_index is None:
                class_index = len(classes) + 1
                classes[word_class_str] = class_index
            word_classes[word] = class_index
        return word_classes

    def _load_head_distortion_table(self, align_model: str) -> Dict[Tuple[int, int], Dict[int, float]]:
        table: Dict[Tuple[int, int], Dict[int, float]] = {}
        for line in load_corpus(self.model_dir / f"src_trg_{align_model}.d4.final"):
            fields = line.split()
            trg_word_class = int(fields[3])
            src_word_class = int(fields[4])
            key = (src_word_class, trg_word_class)
            probs = table.get(key)
            if probs is None:
                probs = {}
                table[key] = probs
            for index, prob_str in enumerate(fields[9:]):
                if prob_str != "0":
                    dj = index - MAX_SENT_LENGTH
                    probs[dj] = float(prob_str)
        return table

    def _load_nonhead_distortion_table(self, align_model: str) -> Dict[int, Dict[int, float]]:
        table: Dict[int, Dict[int, float]] = {}
        ext = "db4" if platform.system() == "Windows" else "D4"
        is_key_line = True
        trg_word_class = 0
        for line in load_corpus(self.model_dir / f"src_trg_{align_model}.{ext}.final"):
            fields = line.split()
            if is_key_line:
                trg_word_class = int(fields[3])
            else:
                probs = table.get(trg_word_class)
                if probs is None:
                    probs = {}
                    table[trg_word_class] = probs
                for index, prob_str in enumerate(fields):
                    if prob_str != "0":
                        dj = index - MAX_SENT_LENGTH
                        probs[dj] = float(prob_str)
            is_key_line = not is_key_line
        return table
