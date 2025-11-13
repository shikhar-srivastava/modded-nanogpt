#!/bin/bash
# Log-parameterized RMSNorm with FP32 precision and functional depth schedule
# Uses w_log parameters in FP32 for precise gradient accumulation
# exp(w_log) computed in FP32, then multiplication in FP32 following large-activations pattern
# Most precise mode for stable training
torchrun --standalone --nproc_per_node=4 train_gpt.py --log-rmsnorm --rmsnorm-fp32 --functional-depth-schedule
