#!/bin/bash
MAX_CONCURRENT=4
TOTAL_GPU=4
RUNNING=0 
base_dir="examples/Eg1"
mkdir -p "${base_dir}/log"
currentgpu=0
randomseeds=($(seq 0 19)) 
nodess=(128 1024)
Ntargets=(25 50 100)

# generate data
for randomseed in "${randomseeds[@]}"; do 
  timestamp=$(date +"%Y%m%d_%H%M%S")
  python "$base_dir/code/E1_gd.py" $currentgpu $randomseed \
  > "$base_dir/log/${timestamp}t_${randomseed}.log" 2>&1 &
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
for nodes in "${nodess[@]}"; do
for randomseed in "${randomseeds[@]}"; do 
  timestamp=$(date +"%Y%m%d_%H%M%S")
  python "$base_dir/code/E1_mse.py" $currentgpu $nodes $randomseed \
  > "$base_dir/log/${timestamp}t_${nodes}__${randomseed}.log" 2>&1 &
  echo "Start PID: $!  Run $currentgpu nodes $nodes $randomseed at $timestamp"
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
for nodes in "${nodess[@]}"; do
for Ntarget in "${Ntargets[@]}"; do
for randomseed in "${randomseeds[@]}"; do
  initid=$((randomseed))
  timestamp=$(date +"%Y%m%d_%H%M%S")

  python "$base_dir/code/E1_tr.py" \
  $currentgpu $nodes $Ntarget $randomseed $initid \
  > "$base_dir/log/${timestamp}t_${nodes}_${Ntarget}_${randomseed}_${initid}.log" 2>&1 &
  echo "Start PID: $!  Run $currentgpu $nodes $Ntarget $randomseed $initid at $timestamp"
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
done

while [ "$RUNNING" -gt 0 ]; do
  wait -n
  echo "Process completed with exit status $?"
  RUNNING=$((RUNNING - 1))
done

