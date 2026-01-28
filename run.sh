#!/bin/bash
BUILD_DIR=/home/heejin5178/hnsw-AMX/hnswlib/build
OUTPUT_DIR=/home/heejin5178/hnsw-AMX/hnswlib/outputs
DURATION_SECONDS=30  # Set to 0 for fixed query count mode, >0 for QPS benchmark mode

# Core Pinning 설정 (ON/OFF 쉽게 전환)
# "on"  = 물리 코어에 pinning (하이퍼스레딩 피함)
# "off" = pinning 없음 (기본 OpenMP 동작)
CORE_PINNING="off"

if [[ "$CORE_PINNING" == "on" ]]; then
  PINNING_OPTS="OMP_PROC_BIND=close OMP_PLACES=cores"
  echo "Core Pinning: ENABLED"
else
  PINNING_OPTS=""
  echo "Core Pinning: DISABLED"
fi

# Thread counts to test
thread_counts=(
  1
  #2
  #4
  #8
  16
  #32
  64
  128
  256
)

# Batch sizes to test
batch_sizes=(
  "-1"
  #1
  #4
  #8
  #16
  #32
  #64
  #128
  #256
)

# Function to drop page cache
drop_cache() {
    sync
    echo 3 | sudo tee /proc/sys/vm/drop_caches > /dev/null
    echo "Page cache dropped"
}
rm -rf ${BUILD_DIR}
mkdir ${BUILD_DIR}
cd ${BUILD_DIR}
cmake .. -DCMAKE_BUILD_TYPE=Release -DCMAKE_CXX_FLAGS="-DDISABLE_AMX_LOG=1"
make all -j32
#./main

compile_types=(
  #"naive"
  #"avx2"
  #"avx512"

  #"naive_mt"
  #"avx2_mt"
  #"avx512_mt"
  #"naive_mt_hnsw"      # HierarchicalNSW with no SIMD
  #"avx2_mt_hnsw"       # HierarchicalNSW with AVX2
  #"avx512_mt_hnsw"     # HierarchicalNSW with AVX512
  #"naive_batch"        # Query batching without SIMD
  #"avx2_batch"         # Query batching with AVX2
  #"avx512_batch"       # Query batching with AVX512
)

workload_types=(
  "A"
  #"B"
  #"C"
  #"D"
)

for compile_type in "${compile_types[@]}"
do
  for wk in "${workload_types[@]}"
  do
    for num_threads in "${thread_counts[@]}"
    do
      for batch_size in "${batch_sizes[@]}"
      do
        if [[ ! -d ${OUTPUT_DIR} ]]; then
          mkdir -p ${OUTPUT_DIR}
        fi
        drop_cache
        echo "Running workload ${wk} with ${compile_type} (threads=${num_threads}, duration=${DURATION_SECONDS}s, batch=${batch_size}).."
        #perf stat -e context-switches,cpu-migrations,page-faults,cycles,instructions,task-clock \
        #  -- env $PINNING_OPTS OMP_NUM_THREADS=${num_threads} DURATION_SECONDS=${DURATION_SECONDS} QUERY_BATCH_SIZE=${batch_size} \
        #  ./main_${compile_type} ../tests/cpp/sift_config_${wk}.txt \
        #${OUTPUT_DIR}/output_${compile_type}_${wk}_thread_${num_threads}_dur_${DURATION_SECONDS}_batch_${batch_size}_$(date +"%m-%d-%H-%M-%S-%3N")
        env $PINNING_OPTS OMP_NUM_THREADS=${num_threads} DURATION_SECONDS=${DURATION_SECONDS} QUERY_BATCH_SIZE=${batch_size} ./main_${compile_type} ../tests/cpp/sift_config_${wk}.txt ${OUTPUT_DIR}/output_${compile_type}_${wk}_thread_${num_threads}_dur_${DURATION_SECONDS}_batch_${batch_size}_$(date +"%m-%d-%H-%M-%S-%3N")
      done
    done
  done
done
echo "Done (non-AMX)!"

### FOR AMX
cd /home/heejin5178/hnsw-AMX/hnswlib
rm -rf ${BUILD_DIR}
mkdir ${BUILD_DIR}
cd ${BUILD_DIR}
cmake -DAMXEnable=ON .. -DCMAKE_BUILD_TYPE=Release -DCMAKE_CXX_FLAGS="-DDISABLE_AMX_LOG=1"
make all -j32

amx_compile_types=(
  #"amx"
  #"amx_bf16"
  #"amx_mt"
  #"amx_bf16_mt"              # 7 tiles (Tile 0-6), single accumulator
  #"amx_mt_hnsw"             # HierarchicalNSW with AMX (FP32 storage)
  "amx_bf16_mt_hnsw"        # HierarchicalNSW with AMX (BF16 storage)
  #"amx_bf16_dual"           # 8 tiles (Tile 0-7), dual accumulator, single-threaded
  #"amx_bf16_dual_mt"        # 8 tiles (Tile 0-7), dual accumulator, multi-threaded
  #"amx_bf16_minimal"        # 3 tiles (Tile 0,1,2), single-threaded
  #"amx_bf16_minimal_mt"     # 3 tiles (Tile 0,1,2), multi-threaded
  #"amx_batch"               # Query batching with AMX (FP32 storage)
  #"amx_bf16_batch"          # Query batching with AMX (BF16 storage)
)

for compile_type in "${amx_compile_types[@]}"
do
  for wk in "${workload_types[@]}"
  do
    for num_threads in "${thread_counts[@]}"
    do
      for batch_size in "${batch_sizes[@]}"
      do
        if [[ ! -d ${OUTPUT_DIR} ]]; then
          mkdir -p ${OUTPUT_DIR}
        fi
        drop_cache
        echo "Running workload ${wk} with ${compile_type} (threads=${num_threads}, duration=${DURATION_SECONDS}s, batch=${batch_size}).."
        env $PINNING_OPTS OMP_NUM_THREADS=${num_threads} DURATION_SECONDS=${DURATION_SECONDS} QUERY_BATCH_SIZE=${batch_size} ./main_${compile_type} ../tests/cpp/sift_config_${wk}.txt ${OUTPUT_DIR}/output_${compile_type}_${wk}_thread_${num_threads}_dur_${DURATION_SECONDS}_batch_${batch_size}_$(date +"%m-%d-%H-%M-%S-%3N")
        #perf stat -e context-switches,cpu-migrations,page-faults,cycles,instructions,task-clock \
        #  -- env $PINNING_OPTS OMP_NUM_THREADS=${num_threads} DURATION_SECONDS=${DURATION_SECONDS} QUERY_BATCH_SIZE=${batch_size} \
        #  ./main_${compile_type} ../tests/cpp/sift_config_${wk}.txt \
        #  ${OUTPUT_DIR}/output_${compile_type}_${wk}_thread_${num_threads}_dur_${DURATION_SECONDS}_batch_${batch_size}_$(date +"%m-%d-%H-%M-%S-%3N")
      done
    done
  done
done
echo "Done (AMX)!"
