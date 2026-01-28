# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is **hnswlib** - a header-only C++ library implementing the Hierarchical Navigable Small World (HNSW) algorithm for fast approximate nearest neighbor search. This fork extends the original with **Intel AMX (Advanced Matrix Extensions)** support for accelerated BF16 operations.

## Build Commands

### Python Bindings

```bash
# Install from source
pip install .

# Run Python tests
make test
# or
python -m unittest discover --start-directory tests/python --pattern "bindings_test*.py"

# Build source distribution
make dist

# Clean build artifacts
make clean
```

### C++ Build (CMake)

```bash
mkdir build && cd build
cmake ..
make

# To enable AMX support, set AMXEnable in CMakeLists.txt or pass it:
cmake -DAMXEnable=true ..
```

### Running C++ Tests

After building, from the `build` directory:
```bash
./test_updates           # Run without updates
./test_updates update    # Run with updates
./searchKnnCloserFirst_test
./searchKnnWithFilter_test
./multiThreadLoad_test
./multiThread_replace_test
./multivector_search_test
./epsilon_search_test
```

For the 200M SIFT test:
```bash
python tests/cpp/download_bigann.py  # Download dataset first
python tests/cpp/update_gen_data.py  # Generate test data
./main
```

## Architecture

### Core Header Files (`hnswlib/`)

- **`hnswlib.h`** - Main entry point. Defines `SpaceInterface`, `AlgorithmInterface`, and SIMD capability detection (SSE/AVX/AVX512/AMX). Includes all other headers.

- **`hnswalg.h`** - Core `HierarchicalNSW` class implementing the HNSW algorithm. Key methods:
  - `addPoint()` - Insert elements (supports updates if label exists)
  - `searchKnn()` - K-nearest neighbor search
  - `searchBaseLayerST()` - Base layer search with AMX batch distance computation when enabled
  - `markDelete()`/`unmarkDelete()` - Soft deletion support

- **`space_l2.h`** - L2 (Euclidean) distance implementations with SIMD variants (SSE/AVX/AVX512)

- **`space_ip.h`** - Inner product distance implementations. Contains:
  - Standard SIMD implementations (SSE/AVX/AVX512)
  - **AMX batch distance functions** (`InnerProductBatchExtAMX`, `amx_inner_product_matrix_fp32`)
  - **BF16 support** (`Bf16InnerProductSpace`, `InnerProductDistanceBf16AVX512`)
  - `enable_amx()` - Linux syscall to enable AMX tile data feature

- **`bruteforce.h`** - Brute-force search baseline implementation

- **`visited_list_pool.h`** - Thread-local visited node tracking for search

- **`stop_condition.h`** - Custom search termination conditions (epsilon search, multi-vector)

### AMX Integration

When `USE_AMX` is defined:
1. `AMXDISTFUNC` typedef enables batch distance computation
2. `HierarchicalNSW::amxdistfunc_` stores the AMX distance function
3. `searchKnn()` and `searchBaseLayerST()` use AMX for batch neighbor distance computation
4. AMX tile configuration uses 16x16 batch sizes for matrix operations

### Python Bindings

- **`python_bindings/bindings.cpp`** - pybind11 bindings exposing `Index` class
- **`python_bindings/LazyIndex.py`** - Lazy-loading index wrapper

### Distance Spaces

| Space | Parameter | Distance |
|-------|-----------|----------|
| L2Space | 'l2' | Squared Euclidean |
| InnerProductSpace | 'ip' | 1 - dot product |
| Bf16InnerProductSpace | - | BF16 inner product (C++ only) |

## Key Algorithm Parameters

- **M** - Max bi-directional links per element (typically 12-48, affects memory ~M*8-10 bytes/element)
- **ef_construction** - Build-time search width (higher = better quality, slower build)
- **ef** - Query-time search width (higher = better recall, slower search, must be >= k)
