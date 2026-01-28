cmake_minimum_required(VERSION 3.0...3.26)

project(hnswlib
    LANGUAGES CXX)

include(GNUInstallDirs)
include(CheckCXXCompilerFlag)
include(CheckCXXSourceCompiles)

option(AMXEnable "Enable Intel AMX support" OFF)
option(DualAccumulator "Enable Dual Accumulator mode for AMX (uses Tile 2 + Tile 7)" OFF)
add_library(hnswlib INTERFACE)
add_library(hnswlib::hnswlib ALIAS hnswlib)

target_include_directories(hnswlib INTERFACE
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}>
    $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>)

# Install
install(DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/hnswlib
    DESTINATION ${CMAKE_INSTALL_INCLUDEDIR})

install(TARGETS hnswlib
    EXPORT hnswlibTargets)

install(EXPORT hnswlibTargets
    FILE hnswlibConfig.cmake
    NAMESPACE hnswlib::
    DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/hnswlib)

# Examples and tests
if(CMAKE_PROJECT_NAME STREQUAL PROJECT_NAME)
    option(HNSWLIB_EXAMPLES "Build examples and tests." ON)
else()
    option(HNSWLIB_EXAMPLES "Build examples and tests." OFF)
endif()
if(HNSWLIB_EXAMPLES)
    set(CMAKE_CXX_STANDARD 11)

    if (CMAKE_CXX_COMPILER_ID MATCHES "Clang")
      SET( CMAKE_CXX_FLAGS  "-Ofast -std=c++11 -DHAVE_CXX0X -openmp -fpic -ftree-vectorize" )
      check_cxx_compiler_flag("-march=native" COMPILER_SUPPORT_NATIVE_FLAG)
      if(COMPILER_SUPPORT_NATIVE_FLAG)
        SET( CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -march=native" )
        message("set -march=native flag")
      else()
        check_cxx_compiler_flag("-mcpu=apple-m1" COMPILER_SUPPORT_M1_FLAG)
        if(COMPILER_SUPPORT_M1_FLAG)
          SET( CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -mcpu=apple-m1" )
          message("set -mcpu=apple-m1 flag")
        endif()
      endif()
    elseif (CMAKE_CXX_COMPILER_ID STREQUAL "GNU")
      SET( CMAKE_CXX_FLAGS  "-Ofast -g -lrt -std=c++11 -DHAVE_CXX0X -march=native -fpic -w -fopenmp -ftree-vectorize -ftree-vectorizer-verbose=0 " )
    elseif (CMAKE_CXX_COMPILER_ID STREQUAL "MSVC")
      SET( CMAKE_CXX_FLAGS  "/O2 -DHAVE_CXX0X /W1 /openmp /EHsc" )
    endif()

if(AMXEnable)
    # Check if compiler supports AMX
    set(CMAKE_REQUIRED_FLAGS "-mamx-bf16")
    set(SOURCE_CODE "
    #if !defined(__AMX_BF16__)
    #error \"AMX not supported\"
    #endif
    int main() { return 0; }
    ")

    check_cxx_source_compiles("${SOURCE_CODE}" HAS_AMX_SUPPORT)
    unset(CMAKE_REQUIRED_FLAGS)

    if(HAS_AMX_SUPPORT)
        message(STATUS "Compiler supports AMX - main_amx will be built")
    else()
        message(STATUS "Compiler does NOT support AMX - main_amx will NOT be built")
    endif()
endif()

    # examples
    add_executable(example_search examples/cpp/example_search.cpp)
    target_link_libraries(example_search hnswlib)

    add_executable(example_epsilon_search examples/cpp/example_epsilon_search.cpp)
    target_link_libraries(example_epsilon_search hnswlib)

    add_executable(example_multivector_search examples/cpp/example_multivector_search.cpp)
    target_link_libraries(example_multivector_search hnswlib)

    add_executable(example_filter examples/cpp/example_filter.cpp)
    target_link_libraries(example_filter hnswlib)

    add_executable(example_replace_deleted examples/cpp/example_replace_deleted.cpp)
    target_link_libraries(example_replace_deleted hnswlib)

    add_executable(example_mt_search examples/cpp/example_mt_search.cpp)
    target_link_libraries(example_mt_search hnswlib)

    
    add_executable(example_mt_search_bf16 examples/cpp/example_mt_search_bf16.cpp)
    target_link_libraries(example_mt_search_bf16 hnswlib)

    add_executable(example_mt_filter examples/cpp/example_mt_filter.cpp)
    target_link_libraries(example_mt_filter hnswlib)

    add_executable(example_mt_replace_deleted examples/cpp/example_mt_replace_deleted.cpp)
    target_link_libraries(example_mt_replace_deleted hnswlib)

    # tests
    add_executable(multivector_search_test tests/cpp/multivector_search_test.cpp)
    target_link_libraries(multivector_search_test hnswlib)

    add_executable(epsilon_search_test tests/cpp/epsilon_search_test.cpp)
    target_link_libraries(epsilon_search_test hnswlib)

    add_executable(test_updates tests/cpp/updates_test.cpp)
    target_link_libraries(test_updates hnswlib)

    add_executable(searchKnnCloserFirst_test tests/cpp/searchKnnCloserFirst_test.cpp)
    target_link_libraries(searchKnnCloserFirst_test hnswlib)

    add_executable(searchKnnWithFilter_test tests/cpp/searchKnnWithFilter_test.cpp)
    target_link_libraries(searchKnnWithFilter_test hnswlib)

    add_executable(multiThreadLoad_test tests/cpp/multiThreadLoad_test.cpp)
    target_link_libraries(multiThreadLoad_test hnswlib)

    add_executable(multiThread_replace_test tests/cpp/multiThread_replace_test.cpp)
    target_link_libraries(multiThread_replace_test hnswlib)

    add_executable(main tests/cpp/main.cpp tests/cpp/sift_1b.cpp)
    target_link_libraries(main hnswlib)

    # main_naive: No SIMD vectorization
    add_executable(main_naive tests/cpp/main.cpp tests/cpp/sift_1b.cpp)
    target_compile_definitions(main_naive PRIVATE NO_MANUAL_VECTORIZATION)
    target_link_libraries(main_naive hnswlib)

    # main_avx2: AVX2 only (no AVX512)
    add_executable(main_avx2 tests/cpp/main.cpp tests/cpp/sift_1b.cpp)
    target_compile_options(main_avx2 PRIVATE -mavx2 -mno-avx512f)
    target_link_libraries(main_avx2 hnswlib)

    # main_avx512: AVX512 enabled
    add_executable(main_avx512 tests/cpp/main.cpp tests/cpp/sift_1b.cpp)
    target_compile_options(main_avx512 PRIVATE -mavx512f)
    target_link_libraries(main_avx512 hnswlib)

    # Multi-threaded versions (non-AMX)
    # main_naive_mt: No SIMD, multi-threaded queries
    add_executable(main_naive_mt tests/cpp/main_mt.cpp tests/cpp/sift_1b_mt.cpp)
    target_compile_definitions(main_naive_mt PRIVATE NO_MANUAL_VECTORIZATION)
    target_link_libraries(main_naive_mt hnswlib)

    # main_avx2_mt: AVX2 only, multi-threaded queries
    add_executable(main_avx2_mt tests/cpp/main_mt.cpp tests/cpp/sift_1b_mt.cpp)
    target_compile_options(main_avx2_mt PRIVATE -mavx2 -mno-avx512f)
    target_link_libraries(main_avx2_mt hnswlib)

    # main_avx512_mt: AVX512, multi-threaded queries
    add_executable(main_avx512_mt tests/cpp/main_mt.cpp tests/cpp/sift_1b_mt.cpp)
    target_compile_options(main_avx512_mt PRIVATE -mavx512f)
    target_link_libraries(main_avx512_mt hnswlib)

    # HNSW versions (multi-threaded)
    # main_naive_mt_hnsw: No SIMD, multi-threaded queries, HierarchicalNSW
    add_executable(main_naive_mt_hnsw tests/cpp/main_mt_hnsw.cpp tests/cpp/sift_1b_mt_hnsw.cpp)
    target_compile_definitions(main_naive_mt_hnsw PRIVATE NO_MANUAL_VECTORIZATION)
    target_link_libraries(main_naive_mt_hnsw hnswlib)

    # main_avx2_mt_hnsw: AVX2 only, multi-threaded queries, HierarchicalNSW
    add_executable(main_avx2_mt_hnsw tests/cpp/main_mt_hnsw.cpp tests/cpp/sift_1b_mt_hnsw.cpp)
    target_compile_options(main_avx2_mt_hnsw PRIVATE -mavx2 -mno-avx512f)
    target_link_libraries(main_avx2_mt_hnsw hnswlib)

    # main_avx512_mt_hnsw: AVX512, multi-threaded queries, HierarchicalNSW
    add_executable(main_avx512_mt_hnsw tests/cpp/main_mt_hnsw.cpp tests/cpp/sift_1b_mt_hnsw.cpp)
    target_compile_options(main_avx512_mt_hnsw PRIVATE -mavx512f)
    target_link_libraries(main_avx512_mt_hnsw hnswlib)

    # main_amx: AMX enabled with FP32 storage (requires AMXEnable=ON)
    if (AMXEnable AND HAS_AMX_SUPPORT)
        add_executable(main_amx tests/cpp/main.cpp tests/cpp/sift_1b.cpp)
        target_compile_options(main_amx PRIVATE -mamx-bf16 -mavx512f)
        target_compile_definitions(main_amx PRIVATE USE_AMX=1)
        target_link_libraries(main_amx hnswlib)

        # main_amx_bf16: AMX enabled with BF16 storage
        add_executable(main_amx_bf16 tests/cpp/main.cpp tests/cpp/sift_1b.cpp)
        target_compile_options(main_amx_bf16 PRIVATE -mamx-bf16 -mavx512f)
        target_compile_definitions(main_amx_bf16 PRIVATE USE_AMX=1 USE_AMX_BF16=1)
        target_link_libraries(main_amx_bf16 hnswlib)

        # main_amx_mt: AMX enabled with FP32 storage, multi-threaded queries
        add_executable(main_amx_mt tests/cpp/main_mt.cpp tests/cpp/sift_1b_mt.cpp)
        target_compile_options(main_amx_mt PRIVATE -mamx-bf16 -mavx512f)
        target_compile_definitions(main_amx_mt PRIVATE USE_AMX=1)
        target_link_libraries(main_amx_mt hnswlib)

        # main_amx_bf16_mt: AMX enabled with BF16 storage, multi-threaded queries
        add_executable(main_amx_bf16_mt tests/cpp/main_mt.cpp tests/cpp/sift_1b_mt.cpp)
        target_compile_options(main_amx_bf16_mt PRIVATE -mamx-bf16 -mavx512f)
        target_compile_definitions(main_amx_bf16_mt PRIVATE USE_AMX=1 USE_AMX_BF16=1)
        target_link_libraries(main_amx_bf16_mt hnswlib)

        # HNSW AMX versions
        # main_amx_mt_hnsw: AMX enabled with FP32 storage, multi-threaded queries, HierarchicalNSW
        add_executable(main_amx_mt_hnsw tests/cpp/main_mt_hnsw.cpp tests/cpp/sift_1b_mt_hnsw.cpp)
        target_compile_options(main_amx_mt_hnsw PRIVATE -mamx-bf16 -mavx512f)
        target_compile_definitions(main_amx_mt_hnsw PRIVATE USE_AMX=1 DISABLE_AMX_LOG=1)
        target_link_libraries(main_amx_mt_hnsw hnswlib)

        # main_amx_bf16_mt_hnsw: AMX enabled with BF16 storage, multi-threaded queries, HierarchicalNSW
        add_executable(main_amx_bf16_mt_hnsw tests/cpp/main_mt_hnsw.cpp tests/cpp/sift_1b_mt_hnsw.cpp)
        target_compile_options(main_amx_bf16_mt_hnsw PRIVATE -mamx-bf16 -mavx512f)
        target_compile_definitions(main_amx_bf16_mt_hnsw PRIVATE USE_AMX=1 USE_AMX_BF16=1 DISABLE_AMX_LOG=1)
        target_link_libraries(main_amx_bf16_mt_hnsw hnswlib)

        # Dual Accumulator versions (uses Tile 2 + Tile 7)
        # main_amx_bf16_dual: AMX BF16 with dual accumulator, single-threaded
        add_executable(main_amx_bf16_dual tests/cpp/main.cpp tests/cpp/sift_1b.cpp)
        target_compile_options(main_amx_bf16_dual PRIVATE -mamx-bf16 -mavx512f)
        target_compile_definitions(main_amx_bf16_dual PRIVATE USE_AMX=1 USE_AMX_BF16=1 USE_DUAL_ACCUMULATOR=1)
        target_link_libraries(main_amx_bf16_dual hnswlib)

        # main_amx_bf16_dual_mt: AMX BF16 with dual accumulator, multi-threaded
        add_executable(main_amx_bf16_dual_mt tests/cpp/main_mt.cpp tests/cpp/sift_1b_mt.cpp)
        target_compile_options(main_amx_bf16_dual_mt PRIVATE -mamx-bf16 -mavx512f)
        target_compile_definitions(main_amx_bf16_dual_mt PRIVATE USE_AMX=1 USE_AMX_BF16=1 USE_DUAL_ACCUMULATOR=1)
        target_link_libraries(main_amx_bf16_dual_mt hnswlib)

        # Minimal Tiles versions (uses only 3 tiles: Tile 0=Data, Tile 1=Query, Tile 2=Acc)
        # main_amx_bf16_minimal: AMX BF16 with 3 tiles, single-threaded
        add_executable(main_amx_bf16_minimal tests/cpp/main.cpp tests/cpp/sift_1b.cpp)
        target_compile_options(main_amx_bf16_minimal PRIVATE -mamx-bf16 -mavx512f)
        target_compile_definitions(main_amx_bf16_minimal PRIVATE USE_AMX=1 USE_AMX_BF16=1 USE_MINIMAL_TILES=1)
        target_link_libraries(main_amx_bf16_minimal hnswlib)

        # main_amx_bf16_minimal_mt: AMX BF16 with 3 tiles, multi-threaded
        add_executable(main_amx_bf16_minimal_mt tests/cpp/main_mt.cpp tests/cpp/sift_1b_mt.cpp)
        target_compile_options(main_amx_bf16_minimal_mt PRIVATE -mamx-bf16 -mavx512f)
        target_compile_definitions(main_amx_bf16_minimal_mt PRIVATE USE_AMX=1 USE_AMX_BF16=1 USE_MINIMAL_TILES=1)
        target_link_libraries(main_amx_bf16_minimal_mt hnswlib)

        # Batch query versions (true query batching with AMX)
        # main_amx_batch: AMX with FP32 storage, batch query processing
        add_executable(main_amx_batch tests/cpp/main_batch.cpp tests/cpp/sift_1b_batch.cpp)
        target_compile_options(main_amx_batch PRIVATE -mamx-bf16 -mavx512f)
        target_compile_definitions(main_amx_batch PRIVATE USE_AMX=1)
        target_link_libraries(main_amx_batch hnswlib)

        # main_amx_bf16_batch: AMX with BF16 storage, batch query processing
        add_executable(main_amx_bf16_batch tests/cpp/main_batch.cpp tests/cpp/sift_1b_batch.cpp)
        target_compile_options(main_amx_bf16_batch PRIVATE -mamx-bf16 -mavx512f)
        target_compile_definitions(main_amx_bf16_batch PRIVATE USE_AMX=1 USE_AMX_BF16=1)
        target_link_libraries(main_amx_bf16_batch hnswlib)
    endif()

    # Non-AMX batch versions (for comparison)
    # main_naive_batch: No SIMD, batch query processing
    add_executable(main_naive_batch tests/cpp/main_batch.cpp tests/cpp/sift_1b_batch.cpp)
    target_compile_definitions(main_naive_batch PRIVATE NO_MANUAL_VECTORIZATION)
    target_compile_options(main_naive_batch PRIVATE -mno-avx -mno-avx2 -mno-avx512f)
    target_link_libraries(main_naive_batch hnswlib)

    # main_avx2_batch: AVX2, batch query processing
    add_executable(main_avx2_batch tests/cpp/main_batch.cpp tests/cpp/sift_1b_batch.cpp)
    target_compile_options(main_avx2_batch PRIVATE -mavx2 -mfma -mno-avx512f)
    target_link_libraries(main_avx2_batch hnswlib)

    # main_avx512_batch: AVX512, batch query processing
    add_executable(main_avx512_batch tests/cpp/main_batch.cpp tests/cpp/sift_1b_batch.cpp)
    target_compile_options(main_avx512_batch PRIVATE -mavx512f)
    target_link_libraries(main_avx512_batch hnswlib)
endif()
