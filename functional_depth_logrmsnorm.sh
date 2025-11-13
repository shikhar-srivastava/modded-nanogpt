#!/bin/bash
# Log-parameterized RMSNorm with functional depth scheduling (BF16 w_log)
# LogRMSNorm: Uses w_log parameters where weight = exp(w_log)
# Benefits: More stable gradient updates in low-precision training
# w_log will be cast to BF16 (model dtype)
torchrun --standalone --nproc_per_node=4 train_gpt.py --log-rmsnorm --functional-depth-schedule

