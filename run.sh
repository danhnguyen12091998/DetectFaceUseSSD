#!/usr/bin/env bash
# Filename: run.sh

export LD_PRELOAD=/home/jetson/.local/lib/python3.6/site-packages/sklearn/__check_build/../../scikit_learn.libs/libgomp-d22c30c5.so.1.0.0

python3 trt_ssd.py --model ssd_mobilenet_v1_face --vid 0 --usb

