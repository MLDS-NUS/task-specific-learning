#!/bin/bash
MAX_CONCURRENT=4
TOTAL_GPU=4
RUNNING=0 
base_dir="examples/Eg1"
mkdir -p "${base_dir}/log"
currentgpu=0
randomseeds=($(seq 0 19)) 

for randomseed in "${randomseeds[@]}"; do 
  timestamp=$(date +"%Y%m%d_%H%M%S")
  python "$base_dir/code/E1_re_mse.py" $currentgpu $randomseed  \
  > "$base_dir/log/${timestamp}t_${randomseed}_${alpha}.log" 2>&1 &
  echo "Start PID: $!  Run GPU $currentgpu $randomseed at $timestamp"
  sleep 1

  currentgpu=$(( (currentgpu + 1) % TOTAL_GPU  )) 
  RUNNING=$((RUNNING + 1)) 
  if [ "$RUNNING" -ge "$MAX_CONCURRENT" ]; then 
    wait -n
    echo "Process completed with exit status $?"
    RUNNING=$((RUNNING - 1))  
  fi
done
