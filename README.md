# Giza-py: MGIZA++ Command-line Runner

giza-py is a simple, Python-based, command-line runner for MGIZA++, a popular tool for building word alignment models.

## Installation

### Python

Giza-py requires [Python 3.7](https://www.python.org/downloads/) or greater.

### .NET Core SDK

Giza-py requires that the [.NET Core 3.1 SDK](https://dotnet.microsoft.com/download) is installed.

### Giza-py

To install Giza-py, simply clone the repo:

```
git clone https://github.com/sillsdev/giza-py.git
```

### MGIZA++

In order to install MGIZA++ on Linux/macOS, follow these steps:

1. Download the [Boost C++ library](https://www.boost.org/) and unzip it.
2. Build Boost:

```
cd <boost_dir>
./bootstrap.sh --prefix=./build --with-libraries=thread,system
./b2 install
```

3. Clone the MGIZA++ repo:

```
git clone https://github.com/moses-smt/mgiza.git
```

4. Build MGIZA++ (CMake is required):

```
cd <mgiza_dir>/mgizapp
cmake -DBOOST_ROOT=<boost_dir>/build -DBoost_USE_STATIC_LIBS=ON -DCMAKE_INSTALL_PREFIX=<giza-py_dir>/.bin .
make
make install
```

## Usage

### Generating alignments

To generate alignments using MGIZA++, run the following command:

```
python3 giza.py --source <src_path> --target <trg_path> --alignments <output_path>
```

The source and target corpora files must be text files where tokens are separated by spaces. Giza-py will output the alignments in Pharaoh format.

Alignment probabilties for each aligned word pair can be output by using the `--include-probs` argument. Giza-py will include alignment probabilities in the generated alignment file. The probabilities are separated from each word pair using a colon `:` delimiter. Here is an example of the Pharaoh format with probabilities included:

```
7-0:0.22661511 5-3:0.4715056 3-6:0.67267063 1-7:0.10234439
0-0:0.75820181 4-1:0.24716581 8-4:0.72411429
```

_Note: The probabilities included in the alignment file are only alignment probabilities and do not include translation probabilities. If you want translation probabilties, they can be obtained by [generating a lexicon](#generating-a-lexicon)._

### Models

By default, Giza-py will generate alignments using the IBM-4 model. To specify a different model, use the `--model` argument.

```
python3 giza.py --source <src_path> --target <trg_path> --alignments <output_path> --model hmm
```

The number of iterations for each stage of training can be specified using the `--m{model_number}` arguments. The following example will train an IBM-4 model with 10 iterations for the IBM-1 stage:

```
python3 giza.py --source <src_path> --target <trg_path> --alignments <output_path> --m1 10
```

The following are the parameters for configuring the number of iterations for each supported model:

- ibm1
  - m1: IBM-1 (default: 5 iterations)
- ibm2
  - m1: IBM-1 (default: 5 iterations)
  - m2: IBM-2 (default: 5 iterations)
- hmm
  - m1: IBM-1 (default: 5 iterations)
  - mh: HMM (default: 5 iterations)
- ibm3
  - m1: IBM-1 (default: 5 iterations)
  - mh: HMM (default: 5 iterations)
  - m3: IBM-3 (default: 5 iterations)
- ibm4
  - m1: IBM-1 (default: 5 iterations)
  - mh: HMM (default: 5 iterations)
  - m3: IBM-3 (default: 5 iterations)
  - m4: IBM-4 (default: 5 iterations)

### Symmetrization

Giza-py generates symmetrized alignments using direct and inverse alignment models. By default, Giza-py will symmetrize alignments using the "grow-diag-final-and" heuristic. To specify a different heuristic, use the `--sym-heuristic` argument.

```
python3 giza.py --source <src_path> --target <trg_path> --alignments <output_path> --sym-heuristic intersection
```

Giza-py supports many different symmetrization heuristics:

- union
- intersection
- och
- grow
- grow-diag
- grow-diag-final
- grow-diag-final-and

### Generating a lexicon

Giza-py can also extract a bilingual lexicon from the trained alignment model.

```
python3 giza.py --source <src_path> --target <trg_path> --lexicon <output_path>
```

The lexicon is extracted as a tab-separated text file. The score for each word pair is the maximum probability from the direct and inverse alignment model.

The lexicon can be filtered by using the `--lexicon-threshold` argument. Giza-py will filter out all translations with a probability that is less than or equal to the specified threshold.
