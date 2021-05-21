# Giza-py: MGIZA++ Command-line Runner
giza-py is a simple, Python-based, command-line runner for MGIZA++, a popular tool for building word alignment models.

## Installation

### Python
Giza-py requires [Python 3.7](https://www.python.org/downloads/) or greater.

### .NET Core SDK
Giza-py requires that the [.NET Core 3.1 SDK](https://dotnet.microsoft.com/download) is installed.

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

### Models
By default, Giza-py will generate alignments using the IBM-4 model. To specify a different model, use the `--model` argument.
```
python3 giza.py --source <src_path> --target <trg_path> --alignments <output_path> --model hmm
```
The following models are supported:
- ibm1
    - IBM-1: 5 iterations
- ibm2
    - IBM-1: 5 iterations
    - IBM-2: 5 iterations
- hmm
    - IBM-1: 5 iterations
    - HMM: 5 iterations
- ibm3
    - IBM-1: 5 iterations
    - HMM: 5 iterations
    - IBM-3: 5 iterations
- ibm4
    - IBM-1: 5 iterations
    - HMM: 5 iterations
    - IBM-3: 5 iterations
    - IBM-4: 5 iterations

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
The score for each word pair is the maximum probability from the direct and inverse alignment model.