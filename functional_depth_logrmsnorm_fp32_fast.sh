#!/bin/bash
# Log-parameterized RMSNorm with FP32 parameters but fast fused kernel
# Uses w_log parameters in FP32 for gradient precision
# exp(w_log) computed in BF16 to enable fused RMSNorm kernel (fastest)
# Balance between speed and FP32 gradient accumulation
torchrun --standalone --nproc_per_node=4 train_gpt.py --log-rmsnorm --rmsnorm-fp32 --rmsnorm-fp32-fast --functional-depth-schedule

