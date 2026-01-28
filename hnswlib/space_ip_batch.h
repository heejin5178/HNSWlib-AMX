#pragma once

#include "hnswlib.h"
#include <immintrin.h>
#include <cstring>

namespace hnswlib {

#if defined(USE_AMX)

// Thread-local buffers to avoid repeated allocation (방향 1)
struct AMXBatchBuffers {
    alignas(64) uint16_t query_chunk[16 * 32];
    alignas(64) uint16_t db_paired[16 * 32];
    alignas(64) float result_tile[16 * 16];
    alignas(64) uint16_t padded_queries[16 * 1024];  // max dim 1024
    alignas(64) uint16_t padded_db[16 * 1024];
    alignas(64) float block_results[16 * 16];
    bool initialized = false;
};

inline AMXBatchBuffers& getAMXBuffers() {
    thread_local AMXBatchBuffers buffers;
    return buffers;
}

// AMX tile config - initialized once per thread
// Uses enable_amx() from space_ip.h for permission request
inline void initAMXTileConfig() {
    // 먼저 AMX 권한 요청 (enable_amx() from space_ip.h)
    static thread_local bool amx_perm_requested = false;
    if (!amx_perm_requested) {
        enable_amx();  // Use existing function from space_ip.h
        amx_perm_requested = true;
    }

    struct __tile_config {
        uint8_t palette_id;
        uint8_t start_row;
        uint8_t reserved_0[14];
        uint16_t colsb[16];
        uint8_t rows[16];
    };

    __tile_config config = {0};
    config.palette_id = 1;
    config.rows[0] = 16;    // Tile 0: queries
    config.colsb[0] = 64;   // 32 BF16 = 64 bytes
    config.rows[1] = 16;    // Tile 1: db vectors
    config.colsb[1] = 64;
    config.rows[2] = 16;    // Tile 2: accumulator
    config.colsb[2] = 64;   // 16 FP32 = 64 bytes

    _tile_loadconfig(&config);
}

// Batch inner product: 16 queries x 16 DB vectors = 256 inner products
// Optimized version with reduced copies (방향 1 + 3)
static void InnerProductBatchAMX16x16(
    const uint16_t* queries,      // 16 x dim (BF16)
    const uint16_t* db_vectors,   // 16 x dim (BF16)
    float* results,               // 16 x 16 output
    size_t dim) {

    AMXBatchBuffers& buf = getAMXBuffers();

    initAMXTileConfig();
    _tile_zero(2);

    for (size_t d = 0; d < dim; d += 32) {
        size_t chunk_size = (d + 32 <= dim) ? 32 : (dim - d);

        // 방향 3: 쿼리 청크 직접 복사 (한 번에)
        for (int q = 0; q < 16; q++) {
            memcpy(&buf.query_chunk[q * 32], &queries[q * dim + d], chunk_size * sizeof(uint16_t));
            // 나머지 패딩 (chunk_size < 32인 경우만)
            if (chunk_size < 32) {
                memset(&buf.query_chunk[q * 32 + chunk_size], 0, (32 - chunk_size) * sizeof(uint16_t));
            }
        }

        // 방향 3: DB 벡터를 바로 AMX 포맷으로 변환 (중간 버퍼 제거)
        // tdpbf16ps B 타일 포맷: 각 열이 하나의 DB 벡터의 chunk
        for (int col = 0; col < 16; col++) {
            const uint16_t* src = &db_vectors[col * dim + d];
            uint16_t* dst = &buf.db_paired[col * 32];
            // 연속 복사 (가장 빠름)
            memcpy(dst, src, chunk_size * sizeof(uint16_t));
            if (chunk_size < 32) {
                memset(dst + chunk_size, 0, (32 - chunk_size) * sizeof(uint16_t));
            }
        }

        _tile_loadd(0, buf.query_chunk, 64);
        _tile_loadd(1, buf.db_paired, 64);
        _tile_dpbf16ps(2, 0, 1);
    }

    // 결과를 직접 출력 버퍼에 저장
    _tile_stored(2, results, 64);

    _tile_release();
}

// Optimized version: directly access DB vectors without intermediate copy
static void InnerProductBatchAMX16x16Direct(
    const uint16_t* queries,           // 16 x dim (BF16), contiguous
    const uint16_t* const* db_ptrs,    // 16 pointers to DB vectors
    float* results,                    // 16 x 16 output
    size_t dim) {

    AMXBatchBuffers& buf = getAMXBuffers();

    initAMXTileConfig();
    _tile_zero(2);

    for (size_t d = 0; d < dim; d += 32) {
        size_t chunk_size = (d + 32 <= dim) ? 32 : (dim - d);

        // 쿼리는 연속이므로 직접 복사
        for (int q = 0; q < 16; q++) {
            memcpy(&buf.query_chunk[q * 32], &queries[q * dim + d], chunk_size * sizeof(uint16_t));
            if (chunk_size < 32) {
                memset(&buf.query_chunk[q * 32 + chunk_size], 0, (32 - chunk_size) * sizeof(uint16_t));
            }
        }

        // DB 포인터에서 직접 읽어서 타일 포맷으로 변환
        for (int col = 0; col < 16; col++) {
            memcpy(&buf.db_paired[col * 32], &db_ptrs[col][d], chunk_size * sizeof(uint16_t));
            if (chunk_size < 32) {
                memset(&buf.db_paired[col * 32 + chunk_size], 0, (32 - chunk_size) * sizeof(uint16_t));
            }
        }

        _tile_loadd(0, buf.query_chunk, 64);
        _tile_loadd(1, buf.db_paired, 64);
        _tile_dpbf16ps(2, 0, 1);
    }

    _tile_stored(2, results, 64);
    _tile_release();
}

// Flexible batch inner product for arbitrary batch sizes
// Optimized with buffer reuse
static void InnerProductBatchAMX(
    const uint16_t* queries,      // num_queries x dim (BF16)
    const uint16_t* db_vectors,   // num_db x dim (BF16)
    float* results,               // num_queries x num_db output
    size_t num_queries,
    size_t num_db,
    size_t dim) {

    AMXBatchBuffers& buf = getAMXBuffers();

    // Process in 16x16 blocks
    for (size_t q_start = 0; q_start < num_queries; q_start += 16) {
        for (size_t db_start = 0; db_start < num_db; db_start += 16) {
            size_t q_end = std::min(q_start + 16, num_queries);
            size_t db_end = std::min(db_start + 16, num_db);
            size_t q_count = q_end - q_start;
            size_t db_count = db_end - db_start;

            // 버퍼 초기화 (필요한 부분만)
            if (q_count < 16) {
                memset(buf.padded_queries, 0, 16 * dim * sizeof(uint16_t));
            }
            if (db_count < 16) {
                memset(buf.padded_db, 0, 16 * dim * sizeof(uint16_t));
            }

            // 쿼리 복사
            for (size_t q = 0; q < q_count; q++) {
                memcpy(&buf.padded_queries[q * dim], &queries[(q_start + q) * dim], dim * sizeof(uint16_t));
            }
            // DB 복사
            for (size_t d = 0; d < db_count; d++) {
                memcpy(&buf.padded_db[d * dim], &db_vectors[(db_start + d) * dim], dim * sizeof(uint16_t));
            }

            // Compute 16x16 block
            InnerProductBatchAMX16x16(buf.padded_queries, buf.padded_db, buf.block_results, dim);

            // Copy valid results
            for (size_t q = 0; q < q_count; q++) {
                for (size_t d = 0; d < db_count; d++) {
                    results[(q_start + q) * num_db + (db_start + d)] = buf.block_results[q * 16 + d];
                }
            }
        }
    }
}

#endif // USE_AMX

// Scalar fallback for batch inner product
static void InnerProductBatchScalar(
    const uint16_t* queries,
    const uint16_t* db_vectors,
    float* results,
    size_t num_queries,
    size_t num_db,
    size_t dim) {

    auto bf16_to_float = [](uint16_t bf16) -> float {
        uint32_t tmp = static_cast<uint32_t>(bf16) << 16;
        return *reinterpret_cast<float*>(&tmp);
    };

    for (size_t q = 0; q < num_queries; q++) {
        for (size_t d = 0; d < num_db; d++) {
            float sum = 0.0f;
            for (size_t i = 0; i < dim; i++) {
                float qv = bf16_to_float(queries[q * dim + i]);
                float dv = bf16_to_float(db_vectors[d * dim + i]);
                sum += qv * dv;
            }
            results[q * num_db + d] = sum;
        }
    }
}

// FP32 version of batch inner product (scalar)
static void InnerProductBatchScalarFP32(
    const float* queries,
    const float* db_vectors,
    float* results,
    size_t num_queries,
    size_t num_db,
    size_t dim) {

    for (size_t q = 0; q < num_queries; q++) {
        for (size_t d = 0; d < num_db; d++) {
            float sum = 0.0f;
            for (size_t i = 0; i < dim; i++) {
                sum += queries[q * dim + i] * db_vectors[d * dim + i];
            }
            results[q * num_db + d] = sum;
        }
    }
}

#if defined(__AVX512F__)
// AVX512 version of batch inner product for FP32
static void InnerProductBatchAVX512FP32(
    const float* queries,
    const float* db_vectors,
    float* results,
    size_t num_queries,
    size_t num_db,
    size_t dim) {

    for (size_t q = 0; q < num_queries; q++) {
        const float* query = queries + q * dim;
        for (size_t d = 0; d < num_db; d++) {
            const float* db_vec = db_vectors + d * dim;

            __m512 sum = _mm512_setzero_ps();
            size_t i = 0;

            // Process 16 floats at a time
            for (; i + 16 <= dim; i += 16) {
                __m512 q_vec = _mm512_loadu_ps(query + i);
                __m512 d_vec = _mm512_loadu_ps(db_vec + i);
                sum = _mm512_fmadd_ps(q_vec, d_vec, sum);
            }

            float result = _mm512_reduce_add_ps(sum);

            // Handle remainder
            for (; i < dim; i++) {
                result += query[i] * db_vec[i];
            }

            results[q * num_db + d] = result;
        }
    }
}
#endif

#if defined(__AVX__) && !defined(__AVX512F__)
// AVX/AVX2 version of batch inner product for FP32
static void InnerProductBatchAVX2FP32(
    const float* queries,
    const float* db_vectors,
    float* results,
    size_t num_queries,
    size_t num_db,
    size_t dim) {

    for (size_t q = 0; q < num_queries; q++) {
        const float* query = queries + q * dim;
        for (size_t d = 0; d < num_db; d++) {
            const float* db_vec = db_vectors + d * dim;

            __m256 sum = _mm256_setzero_ps();
            size_t i = 0;

            // Process 8 floats at a time
            for (; i + 8 <= dim; i += 8) {
                __m256 q_vec = _mm256_loadu_ps(query + i);
                __m256 d_vec = _mm256_loadu_ps(db_vec + i);
                sum = _mm256_fmadd_ps(q_vec, d_vec, sum);
            }

            // Horizontal sum
            __m128 hi = _mm256_extractf128_ps(sum, 1);
            __m128 lo = _mm256_castps256_ps128(sum);
            __m128 sum128 = _mm_add_ps(hi, lo);
            sum128 = _mm_hadd_ps(sum128, sum128);
            sum128 = _mm_hadd_ps(sum128, sum128);
            float result = _mm_cvtss_f32(sum128);

            // Handle remainder
            for (; i < dim; i++) {
                result += query[i] * db_vec[i];
            }

            results[q * num_db + d] = result;
        }
    }
}
#endif

}  // namespace hnswlib
