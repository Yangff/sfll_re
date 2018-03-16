An auto analysis tool based on the Logic lock on [SFLL-HD benchmark](https://github.com/DfX-NYUAD/CCS17).

## Prequirement
Python3, z3-solver, verilator

## Installation
1. Install python however you want
2. Install [z3](https://github.com/Z3Prover/z3) on Linux (since that benchmark can only runs on linux)
3. Install verilator required by SFLL-HD benchmark

## Run
1. Clone the project with submodule
2. Run the following command, `-i` is bench file and `-e` is oracle.
```sh
python3 main.py -i ./target/benchmarks/sfll_hd/dfx_sfll_k256_h32.bench -e ./target/bin/DfX_64bit
```
3. In fact this whole thing can run without an oracle with some tiny modify.