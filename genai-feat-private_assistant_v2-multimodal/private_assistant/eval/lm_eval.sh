#!/usr/bin/env bash
# Example: tiny harness run against a local model (if supported)
# pip install lm-eval
lm_eval --model dummy --tasks boolq --num_fewshot 0
