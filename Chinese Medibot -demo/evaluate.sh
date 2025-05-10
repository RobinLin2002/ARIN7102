#!/bin/bash

# 最简单的nohup运行脚本
nohup python3 evaluate_mcq.py > evaluation.log 2>&1 &

echo "程序已在后台运行"
echo "日志文件: evaluation.log"
echo "PID: $!"