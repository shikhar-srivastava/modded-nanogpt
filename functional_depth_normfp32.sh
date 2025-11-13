#!/bin/bash
# Standard mode: Precise FP32 RMSNorm computation
# Computes RMSNorm in FP32 then converts back to BF16
# More precise but may be slightly slower than fast mode
# Note: You may see dtype mismatch warnings - these are expected and handled efficiently
torchrun --standalone --nproc_per_node=4 train_gpt.py --rmsnorm-fp32 --functional-depth-schedule
