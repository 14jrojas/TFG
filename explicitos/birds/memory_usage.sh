#!/bin/bash

nvidia-smi --query-compute-apps=pid,gpu_uuid,used_memory --format=csv --id=2,3 -l 5 > ./memory_usage/birds_resnet50.txt &

pid=$!

python resnet50.py

kill $pid
