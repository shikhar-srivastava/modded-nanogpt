#!/bin/bash
# Optimized version using fast RMSNorm mode
# Uses fused kernel by casting FP32 weights to BF16 (faster but slightly less precise)
# Recommended if the dtype mismatch warning is causing performance issues
torchrun --standalone --nproc_per_node=4 train_gpt.py --rmsnorm-fp32 --rmsnorm-fp32-fast --functional-depth-schedule

