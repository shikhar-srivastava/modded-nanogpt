#!/bin/bash
# Log-parameterized RMSNorm with functional depth schedule
# Uses w_log parameters (default BF16) for fastest fused kernel path
# Weight = exp(w_log), computed in BF16 to match activations
# Minimal overhead while maintaining stable low-precision training
torchrun --standalone --nproc_per_node=4 train_gpt.py --log-rmsnorm --functional-depth-schedule
