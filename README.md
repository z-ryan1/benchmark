# PyTorch Benchmarks
This is a collection of open source benchmarks used to evaluate PyTorch performance.

`torchbenchmark/models` contains copies of popular or exemplary workloads which have been modified to
(a) expose a standardized API for benchmark drivers, (b) optionally, enable JIT,
 (c) contain a miniature version of train/test data and a dependency install script.

## Installation
The benchmark suite should be self contained in terms of dependencies,
except for the torch products which are intended to be installed separately so
different torch versions can be benchmarked.

### Using Pre-built Packages
Use python 3.7 as currently there are compatibility issues with 3.8+.  Conda is optional but suggested.  To switch to python 3.7 in conda:
```
# using your current conda environment:
conda install -y python=3.7

# or, using a new conda environment
conda create -n torchbenchmark python=3.7
conda activate torchbenchmark
```

Install pytorch, torchtext, and torchvision using conda:
```
conda install -y pytorch torchtext torchvision -c pytorch-nightly
```
Or use pip:
(but don't mix and match pip and conda for the torch family of libs! - [see notes below](#notes))
```
pip install --pre torch torchvision torchtext -f https://download.pytorch.org/whl/nightly/cu102/torch_nightly.html
```

NOTE: The install file below uses git-lfs, run the following commands to get it working
```
sh gitlfs_extract.sh
export PATH="$PATH:$(pwd):gitlfs_dir"
```

Install the benchmark suite, which will recursively install dependencies for all the models.  Currently, the repo is intended to be installed from the source tree.
```
git clone <benchmark>
cd <benchmark>
python install.py
```

### Running benchmarks
run `sh run_bench.sh`
