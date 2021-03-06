An auto analysis tool based on the Logic lock on [SFLL-HD benchmark](https://github.com/DfX-NYUAD/CCS17).

## Prequirement
Python3, z3-solver, verilator

## Installation
1. Install python3
2. Install [z3](https://github.com/Z3Prover/z3) on Linux (since that benchmark can only run on linux)
3. Install [verilator](https://www.veripool.org/wiki/verilator) required by SFLL-HD benchmark

## Run
1. Clone the project with submodule
```sh
git clone --recurse-submodules https://github.com/Yangff/sfll_re.git
```
2. Run the following command, `-i` is bench file and `-e` is oracle.
```sh
python3 main.py -i ./target/benchmarks/sfll_hd/dfx_sfll_k256_h32.bench -e ./target/bin/DfX_64bit
```
3. The result will give you two possible complementary bitstreams with `H` and `N-H`. If you can set `H`, they are both solutions of the original circuit.  
4. Or, if you don't have any oracle, you can still run this attack without `-e` arguemnt and it will give you the same result as you have the oracle. This is because we can observe perturb node and use it as an oracle. 