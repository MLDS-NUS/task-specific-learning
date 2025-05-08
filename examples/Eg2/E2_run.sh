#!/bin/bash
MAX_CONCURRENT=4
TOTAL_GPU=4
RUNNING=0 
base_dir="examples/Eg2"
mkdir -p "${base_dir}/log"
currentgpu=0
randomseeds=($(seq 0 19)) 
nodess=(8 12 16 24 32)

# generate data
for randomseed in "${randomseeds[@]}"; do
  timestamp=$(date +"%Y%m%d_%H%M%S")
  python "$base_dir/code/E2_gd.py" $currentgpu $randomseed \
  > "$base_dir/log/${timestamp}_${randomseed}_0.log" 2>&1 &
  echo "Start PID: $!  Run $currentgpu $randomseed at $timestamp"
  sleep 1

  currentgpu=$(( (currentgpu + 1) % TOTAL_GPU  )) 
  RUNNING=$((RUNNING + 1)) 
  if [ "$RUNNING" -ge "$MAX_CONCURRENT" ]; then 
    wait -n
    echo "Process completed with exit status $?"
    RUNNING=$((RUNNING - 1))  
  fi
done

# training by minimizing MSE
for randomseed in "${randomseeds[@]}"; do
for nodes in "${nodess[@]}"; do
  timestamp=$(date +"%Y%m%d_%H%M%S")
  python "$base_dir/code/E2_mse.py" $currentgpu $nodes $randomseed \
  > "$base_dir/log/${timestamp}_${nodes}_${randomseed}_0.log" 2>&1 &
  echo "Start PID: $!  Run $currentgpu $nodes $randomseed at $timestamp"
  sleep 1

  currentgpu=$(( (currentgpu + 1) % TOTAL_GPU  )) 
  RUNNING=$((RUNNING + 1)) 
  if [ "$RUNNING" -ge "$MAX_CONCURRENT" ]; then 
    wait -n
    echo "Process completed with exit status $?"
    RUNNING=$((RUNNING - 1))  
  fi
done
done

# training by minimizing TS
nodess=(0 1 2 3 4 5 8 12 16 24 32)
for randomseed in "${randomseeds[@]}"; do
for nodes in "${nodess[@]}"; do
  timestamp=$(date +"%Y%m%d_%H%M%S")

  python "$base_dir/code/E2_tlb.py" $currentgpu $nodes $randomseed \
  > "$base_dir/log/${timestamp}t_${nodes}_${randomseed}.log" 2>&1 &
  echo "Start PID: $!  Run $currentgpu $nodes $randomseed at $timestamp"
  sleep 1

  currentgpu=$(( (currentgpu + 1) % TOTAL_GPU  )) 
  RUNNING=$((RUNNING + 1)) 
  if [ "$RUNNING" -ge "$MAX_CONCURRENT" ]; then 
    wait -n
    echo "Process completed with exit status $?"
    RUNNING=$((RUNNING - 1))  
  fi
done
done


while [ "$RUNNING" -gt 0 ]; do
  wait -n
  echo "Process completed with exit status $?"
  RUNNING=$((RUNNING - 1))
done




