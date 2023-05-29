#!/bin/bash

nvidia-smi --query-compute-apps=pid,used_memory --format=csv --id=1 -l 5 > ./memory_usage/memory_densenet121-cifar.txt &

pid=$!

python densenet121-memory.py

kill $pid
