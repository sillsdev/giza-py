import argparse
import subprocess
import tempfile
from pathlib import Path

from giza_aligner import HmmGizaAligner, Ibm1GizaAligner, Ibm2GizaAligner, Ibm3GizaAligner, Ibm4GizaAligner


def main() -> None:
    parser = argparse.ArgumentParser(description="Aligns the parallel corpus for an experiment")
    parser.add_argument("--bin", type=str, default=".bin", metavar="PATH", help="The mgiza++ folder")
    parser.add_argument("--source", type=str, required=True, metavar="PATH", help="The source corpus")
    parser.add_argument("--target", type=str, required=True, metavar="PATH", help="The target corpus")
    parser.add_argument("--alignments", type=str, default=None, metavar="PATH", help="The output alignments")
    parser.add_argument("--lexicon", type=str, default=None, metavar="PATH", help="The output lexicon")
    parser.add_argument(
        "--model",
        type=str,
        choices=["ibm1", "ibm2", "hmm", "ibm3", "ibm4"],
        default="ibm4",
        help="The word alignment model",
    )
    parser.add_argument(
        "--sym-heuristic",
        type=str,
        choices=["union", "intersection", "och", "grow", "grow-diag", "grow-diag-final", "grow-diag-final-and"],
        default="grow-diag-final-and",
        help="The symmetrization heuristic",
    )
    args = parser.parse_args()

    print("Installing dependencies...", end="", flush=True)
    subprocess.run(["dotnet", "tool", "restore"], stdout=subprocess.DEVNULL)
    print(" done.")

    bin_dir = Path(args.bin)

    model: str = args.model
    model = model.lower()

    with tempfile.TemporaryDirectory() as td:
        temp_dir = Path(td)
        if model == "ibm1":
            aligner = Ibm1GizaAligner(bin_dir, temp_dir)
        elif model == "ibm2":
            aligner = Ibm2GizaAligner(bin_dir, temp_dir)
        elif model == "hmm":
            aligner = HmmGizaAligner(bin_dir, temp_dir)
        elif model == "ibm3":
            aligner = Ibm3GizaAligner(bin_dir, temp_dir)
        elif model == "ibm4":
            aligner = Ibm4GizaAligner(bin_dir, temp_dir)

        source_path = Path(args.source)
        target_path = Path(args.target)
        print("Training...")
        aligner.train(source_path, target_path)

        if args.alignments is not None:
            alignments_path = Path(args.alignments)
            print("Generating alignments...", end="", flush=True)
            aligner.align(alignments_path, args.sym_heuristic)
            print(" done.")
        if args.lexicon is not None:
            lexicon_path = Path(args.lexicon)
            print("Extracting lexicon...", end="", flush=True)
            aligner.extract_lexicon(lexicon_path)
            print(" done.")


if __name__ == "__main__":
    main()
