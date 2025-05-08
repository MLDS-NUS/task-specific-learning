#!/bin/bash
MAX_CONCURRENT=4
TOTAL_GPU=4
RUNNING=0 
base_dir="examples/Eg3"
mkdir -p "${base_dir}/log"
currentgpu=0
nodes=32
meps=(0 25 50 75)
randomseeds=($(seq 0 19)) 
Ds=(0.1 0.2 1.0)

# generate data
python "$base_dir/code/E3_mep.py"  > "$base_dir/log/mep.log" 2>&1 &
echo "Start PID: $!"

for D in "${Ds[@]}"; do
  timestamp=$(date +"%Y%m%d_%H%M%S")
  python "$base_dir/code/E3_gd.py" $currentgpu $D \
  > "$base_dir/log/${timestamp}m_${D}.log" 2>&1 &
  echo "Start PID: $! Run GPU $currentgpu $D at $timestamp"
  sleep 1

  currentgpu=$(( (currentgpu + 1) % TOTAL_GPU  )) 
  RUNNING=$((RUNNING + 1)) 
  if [ "$RUNNING" -ge "$MAX_CONCURRENT" ]; then 
    wait -n
    echo "Process completed with exit status $?"
    RUNNING=$((RUNNING - 1))  
  fi
done

# generate data and training by minimizing MSE
# the data generation will be skipped if already exists
for randomseed in "${randomseeds[@]}"; do
for mep in "${meps[@]}"; do
for D in "${Ds[@]}"; do
  timestamp=$(date +"%Y%m%d_%H%M%S")
  python "$base_dir/code/E3_mse.py" $currentgpu $nodes $mep $randomseed $D \
  > "$base_dir/log/${timestamp}m_${nodes}_${mep}_${randomseed}_${D}.log" 2>&1 &
  echo "Start PID: $! Run GPU $currentgpu nodes $nodes mep $mep $randomseed $D at $timestamp"
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


# training by minimizing TS
for D in "${Ds[@]}"; do
for randomseed in "${randomseeds[@]}"; do
for mep in "${meps[@]}"; do
  timestamp=$(date +"%Y%m%d_%H%M%S")
  python "$base_dir/code/E3_tr.py" $currentgpu $nodes $mep $randomseed $D \
  > "$base_dir/log/${timestamp}t_${nodes}_${mep}_${randomseed}_${D}.log" 2>&1 &
  echo "Start PID: $! Run GPU $currentgpu nodes $nodes mep $mep $randomseed $D at $timestamp"
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



