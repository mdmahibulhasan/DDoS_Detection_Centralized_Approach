#!/bin/bash
module load StdEnv/2023
#module load ~/.venvs/fl_env/bin/~/.venvs/fl_env/bin/python/3.11.5
module load r-bundle-bioconductor/3.18
module load gcc/12.3
module load cuda
module load cudnn
module load arrow
module load perl


echo "Starting client 5.........."

~/.venvs/fl_env/bin/python c5.py > ./logs/client_5.log 2>&1 &
CLIENT5_PID=$!

echo "Starting client 6.........."
~/.venvs/fl_env/bin/python c6.py > ./logs/client_6.log 2>&1 &
CLIENT6_PID=$!