#!/bin/bash
# Log-parameterized RMSNorm with FP32 weights and functional depth scheduling
# LogRMSNorm: Uses w_log parameters where weight = exp(w_log)
# Benefits: More stable gradient updates in low-precision training
# w_log stored in FP32 for precise gradient accumulation
torchrun --standalone --nproc_per_node=4 train_gpt.py --log-rmsnorm --rmsnorm-fp32 --functional-depth-schedule

