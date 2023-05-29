#!/bin/bash

nvidia-smi --query-compute-apps=pid,gpu_uuid,used_memory --format=csv --id=0 -l 5 > ./memory_usage/prueba.txt &

pid=$!

python tools/cls_train.py --cfg ./experiments/cifar/cls_mdeq_LARGE.yaml

kill $pid
