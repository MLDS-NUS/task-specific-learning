#!/bin/bash
MAX_CONCURRENT=4
TOTAL_GPU=4
RUNNING=0 
base_dir="examples/Eg1"
mkdir -p "${base_dir}/log"
currentgpu=0
randomseeds=($(seq 0 19)) 
alphas=(0.0 0.25 0.5 0.75 0.99)

# generate data
for randomseed in "${randomseeds[@]}"; do 
for alpha in "${alphas[@]}"; do 
  timestamp=$(date +"%Y%m%d_%H%M%S")
  python "$base_dir/code/E1_as_gd.py" $currentgpu $randomseed $alpha \
  > "$base_dir/log/${timestamp}t_${randomseed}_${alpha}.log" 2>&1 &
  echo "Start PID: $!  Run GPU $currentgpu $randomseed $alpha at $timestamp"
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

# training by minimizing MSE
for alpha in "${alphas[@]}"; do 
for randomseed in "${randomseeds[@]}"; do 
  timestamp=$(date +"%Y%m%d_%H%M%S")
  python "$base_dir/code/E1_as_mse.py" $currentgpu 128 $randomseed $alpha \
  > "$base_dir/log/${timestamp}t_${randomseed}_${alpha}.log" 2>&1 &
  echo "Start PID: $!  Run GPU $currentgpu $randomseed $alpha at $timestamp"
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
for randomseed in "${randomseeds[@]}"; do 
for alpha in "${alphas[@]}"; do 
  timestamp=$(date +"%Y%m%d_%H%M%S")
  python "$base_dir/code/E1_as_tr.py" $currentgpu 128 $randomseed $alpha \
  > "$base_dir/log/${timestamp}t_${randomseed}_${alpha}.log" 2>&1 &
  echo "Start PID: $!  Run GPU $currentgpu $randomseed $alpha at $timestamp"
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



