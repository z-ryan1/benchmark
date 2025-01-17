#!/bin/bash

set -e
# fix GPU id ordering
export CUDA_DEVICE_ORDER=PCI_BUS_ID

# gpu index
if test -z "$1"
then
    echo Usage: "$0" GPU_IDX 1>&2
    exit 1
fi

gpu_name=$(nvidia-smi --format=csv,noheader --query-gpu=name --id="$1" | sed 's/ /_/g')

export CUDA_VISIBLE_DEVICES="$1"

for i in TF32 FP32
do
    if test "$i" = "TF32"
    then
	unset NVIDIA_TF32_OVERRIDE
	echo running benchmark with "$i"
    else
	export NVIDIA_TF32_OVERRIDE=0
	echo running benchmark with "$i"
    fi
    name="$(hostname -s)_${gpu_name}_$i" 

    pytest test_bench.py --benchmark-save-data -k "cuda-eager" --ignore_machine_config --benchmark-autosave --benchmark-warmup=on --cache-clear --benchmark-min-rounds=500 --benchmark-json="$name".json | tee "$name".out
    python parse.py --file "$name".json --out "$name".csv
done
