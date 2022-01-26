# Vector Benchmarking

Most of the code here is taken from [PyTorch Benchmark](https://github.com/pytorch/benchmark) with some modifications. This benchmark runs a subset of models of the PyTorch benchmark with some additions, namely Seq2Seq, MLP and GAT which we hope to contribute upstream later on.

All benchmarks run on cuda-eager which we believe is most indicative of the workloads of our cluster.

## Installation
We support only python 3.7 in our suite. With the environment being installed using python [venv](https://docs.python.org/3.7/library/venv.html)

## Install the packages with cuda version dependencies
```
# create the venv via python3.7 -m venv env_to_use and then activate it

pip3 install torch==1.10.1+cu113 torchvision==0.11.2+cu113 torchtext -f https://download.pytorch.org/whl/cu113/torch_stable.html
pip3 install torch-scatter -f https://data.pyg.org/whl/torch-1.10.1+cu113.html
pip3 install torch-sparse -f https://data.pyg.org/whl/torch-1.10.1+cu113.html
```

Install the benchmark suite dependencies.  Currently, the repo is intended to be installed from the source tree.
```
git clone <benchmark>
cd <benchmark>
pip install -r requirements.txt
```

## Running our benchmark

```bash
bash run_bench.sh 0
```

This script will then produce .out, .csv, .json files which can be shared
