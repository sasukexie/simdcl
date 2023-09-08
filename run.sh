#!/bin/bash
# conda activate rs
# cd /media/ai/rs/simdcl
# nohup ./run.sh > log/run.log 2>&1 &

# python run_main.py --dataset='ml-1m' --train_batch_size=256 lmd=0.1 --lmd_sem=0.1 --model='DuoRec' --contrast='us_x' --sim='dot' --tau=1

python run_main.py
