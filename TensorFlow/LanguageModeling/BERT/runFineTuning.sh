#!/bin/bash

TF_XLA_FLAGS=--tf_xla_auto_jit=1 ./scripts/run_squad.sh 10 5e-6 fp16 true 8 384 128 base 1.1 data/download/google_pretrained_weights/uncased_L-12_H-768_A-12/bert_model.ckpt 2 &> logs_finetuning/bert_base_xla_fp16_run1.log
TF_XLA_FLAGS=--tf_xla_auto_jit=-1 ./scripts/run_squad.sh 10 5e-6 fp16 false 8 384 128 base 1.1 data/download/google_pretrained_weights/uncased_L-12_H-768_A-12/bert_model.ckpt 2 &> logs_finetuning/bert_base_fp16_run1.log
TF_XLA_FLAGS=--tf_xla_auto_jit=1 ./scripts/run_squad.sh 5 5e-6 fp32 true 8 384 128 base 1.1 data/download/google_pretrained_weights/uncased_L-12_H-768_A-12/bert_model.ckpt 2 &> logs_finetuning/bert_base_xla_fp32_run1.log
TF_XLA_FLAGS=--tf_xla_auto_jit=-1 ./scripts/run_squad.sh 5 5e-6 fp32 false 8 384 128 base 1.1 data/download/google_pretrained_weights/uncased_L-12_H-768_A-12/bert_model.ckpt 2 &> logs_finetuning/bert_base_fp32_run1.log

TF_XLA_FLAGS=--tf_xla_auto_jit=1 ./scripts/run_squad.sh 2 5e-6 fp16 true 8 384 128 large 1.1 data/download/google_pretrained_weights/uncased_L-24_H-1024_A-16/bert_model.ckpt 2 &> logs_finetuning/bert_large_xla_fp16_run1.log
TF_XLA_FLAGS=--tf_xla_auto_jit=-1 ./scripts/run_squad.sh 2 5e-6 fp16 false 8 384 128 large 1.1 data/download/google_pretrained_weights/uncased_L-24_H-1024_A-16/bert_model.ckpt 2 &> logs_finetuning/bert_large_fp16_run1.log
TF_XLA_FLAGS=--tf_xla_auto_jit=1 ./scripts/run_squad.sh 1 5e-6 fp32 true 8 384 128 large 1.1 data/download/google_pretrained_weights/uncased_L-24_H-1024_A-16/bert_model.ckpt 2 &> logs_finetuning/bert_large_xla_fp32_run1.log
TF_XLA_FLAGS=--tf_xla_auto_jit=-1 ./scripts/run_squad.sh 1 5e-6 fp32 false 8 384 128 large 1.1 data/download/google_pretrained_weights/uncased_L-24_H-1024_A-16/bert_model.ckpt 2 &> logs_finetuning/bert_large_fp32_run1.log


TF_XLA_FLAGS=--tf_xla_auto_jit=1 ./scripts/run_glue.sh MRPC 16 5e-6 fp16 true 8 384 128 base 3 0.1 data/download/google_pretrained_weights/uncased_L-12_H-768_A-12/bert_model.ckpt &> logs_finetuning/bert_base_xla_fp16_mrpc_run1.log 
TF_XLA_FLAGS=--tf_xla_auto_jit=1 ./scripts/run_glue.sh MRPC 8 5e-6 fp32 true 8 384 128 base 3 0.1 data/download/google_pretrained_weights/uncased_L-12_H-768_A-12/bert_model.ckpt &> logs_finetuning/bert_base_xla_fp32_mrpc_run1.log 
TF_XLA_FLAGS=--tf_xla_auto_jit=-1 ./scripts/run_glue.sh MRPC 16 5e-6 fp16 false 8 384 128 base 3 0.1 data/download/google_pretrained_weights/uncased_L-12_H-768_A-12/bert_model.ckpt &> logs_finetuning/bert_base_fp16_mrpc_run1.log 
TF_XLA_FLAGS=--tf_xla_auto_jit=-1 ./scripts/run_glue.sh MRPC 8 5e-6 fp32 false 8 384 128 base 3 0.1 data/download/google_pretrained_weights/uncased_L-12_H-768_A-12/bert_model.ckpt &> logs_finetuning/bert_base_fp32_mrpc_run1.log 

