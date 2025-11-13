#!/bin/bash
# Combine Functional Depth Schedule + Layer Norm Scaling
# Tests both optimizations together

torchrun --nproc_per_node=8 train_gpt.py --functional-depth-schedule --lns

