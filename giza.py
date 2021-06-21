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
    parser.add_argument("--probs", type=str, default=None, metavar="PATH", help="The output alignment probabilities")
    parser.add_argument("--lexicon", type=str, default=None, metavar="PATH", help="The output lexicon")
    parser.add_argument(
        "--lexicon-threshold", type=float, default=0.0, metavar="THRESHOLD", help="The lexicon probability threshold"
    )
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
    parser.add_argument("--m1", type=int, default=None, metavar="ITERATIONS", help="The number of IBM-1 iterations")
    parser.add_argument("--m2", type=int, default=None, metavar="ITERATIONS", help="The number of IBM-2 iterations")
    parser.add_argument("--mh", type=int, default=None, metavar="ITERATIONS", help="The number of HMM iterations")
    parser.add_argument("--m3", type=int, default=None, metavar="ITERATIONS", help="The number of IBM-3 iterations")
    parser.add_argument("--m4", type=int, default=None, metavar="ITERATIONS", help="The number of IBM-4 iterations")
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
            aligner = Ibm1GizaAligner(bin_dir, temp_dir, m1=args.m1)
        elif model == "ibm2":
            aligner = Ibm2GizaAligner(bin_dir, temp_dir, m1=args.m1, m2=args.m2)
        elif model == "hmm":
            aligner = HmmGizaAligner(bin_dir, temp_dir, m1=args.m1, mh=args.mh)
        elif model == "ibm3":
            aligner = Ibm3GizaAligner(bin_dir, temp_dir, m1=args.m1, m2=args.m2, mh=args.mh, m3=args.m3)
        elif model == "ibm4":
            aligner = Ibm4GizaAligner(bin_dir, temp_dir, m1=args.m1, m2=args.m2, mh=args.mh, m3=args.m3, m4=args.m4)

        source_path = Path(args.source)
        target_path = Path(args.target)
        print("Training...")
        aligner.train(source_path, target_path)

        if args.alignments is not None:
            alignments_file_path = Path(args.alignments)
            alignment_probs_file_path = None if args.probs is None else Path(args.probs)
            print("Generating alignments...", end="", flush=True)
            aligner.align(alignments_file_path, alignment_probs_file_path, args.sym_heuristic)
            print(" done.")
        if args.lexicon is not None:
            lexicon_path = Path(args.lexicon)
            print("Extracting lexicon...", end="", flush=True)
            aligner.extract_lexicon(lexicon_path, args.lexicon_threshold)
            print(" done.")


if __name__ == "__main__":
    main()
