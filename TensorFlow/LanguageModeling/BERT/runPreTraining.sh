#!/bin/bash

TF_XLA_FLAGS=--tf_xla_auto_jit=1 bash scripts/run_pretraining_lamb.sh 32 8 8 7.5e-4 5e-4 fp16 true 8 1 1 200 100 1 1 base &> logs_pretrain/bert_base_pretrain_xla_fp16.txt
TF_XLA_FLAGS=--tf_xla_auto_jit=1 bash scripts/run_pretraining_lamb.sh 32 8 8 7.5e-4 5e-4 fp32 true 8 1 1 200 100 1 1 base &> logs_pretrain/bert_base_pretrain_xla_fp32.txt
TF_XLA_FLAGS=--tf_xla_auto_jit=1 bash scripts/run_pretraining_lamb.sh 1 1 1 7.5e-4 5e-4 fp32 true 8 1 1 200 100 1 1 large &> logs_pretrain/bert_large_pretrain_xla_fp32.txt
TF_XLA_FLAGS=--tf_xla_auto_jit=1 bash scripts/run_pretraining_lamb.sh 4 2 2 7.5e-4 5e-4 fp16 true 8 1 1 200 100 1 1 large &> logs_pretrain/bert_large_pretrain_xla_fp16.txt

TF_XLA_FLAGS=--tf_xla_auto_jit=-1 bash scripts/run_pretraining_lamb.sh 32 8 8 7.5e-4 5e-4 fp16 false 8 1 1 200 100 1 1 base &> logs_pretrain/bert_base_pretrain_fp16.txt
TF_XLA_FLAGS=--tf_xla_auto_jit=-1 bash scripts/run_pretraining_lamb.sh 32 8 8 7.5e-4 5e-4 fp32 false 8 1 1 200 100 1 1 base &> logs_pretrain/bert_base_pretrain_fp32.txt
TF_XLA_FLAGS=--tf_xla_auto_jit=-1 bash scripts/run_pretraining_lamb.sh 1 1 1 7.5e-4 5e-4 fp32 false 8 1 1 200 100 1 1 large &> logs_pretrain/bert_large_pretrain_fp32.txt
TF_XLA_FLAGS=--tf_xla_auto_jit=-1 bash scripts/run_pretraining_lamb.sh 4 2 2 7.5e-4 5e-4 fp16 false 8 1 1 200 100 1 1 large &> logs_pretrain/bert_large_pretrain_fp16.txt
