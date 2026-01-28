#pragma once

#include "hnswlib.h"
#include "space_ip_batch.h"
#include <vector>
#include <queue>
#include <algorithm>
#include <omp.h>

namespace hnswlib {

// Thread-local buffers for bruteforce batch operations (방향 1)
struct BruteforceBatchBuffers {
    alignas(64) uint16_t db_batch_bf16[16 * 1024];  // max 16 vectors x 1024 dims
    alignas(64) float db_batch_fp32[16 * 1024];
    alignas(64) float batch_dists[16 * 16];
    const uint16_t* db_ptrs_bf16[16];
    const float* db_ptrs_fp32[16];
};

inline BruteforceBatchBuffers& getBFBuffers() {
    thread_local BruteforceBatchBuffers buffers;
    return buffers;
}

// Batch brute force search that processes multiple queries together
// Uses AMX for batch inner product when available
template<typename dist_t>
class BruteforceBatchSearch {
public:
    size_t maxelements_;
    size_t cur_element_count_;
    size_t dim_;
    size_t data_size_;
    char* data_;
    bool is_bf16_;  // true if data is stored as BF16

    BruteforceBatchSearch(size_t dim, size_t maxelements, bool use_bf16 = false)
        : dim_(dim), maxelements_(maxelements), cur_element_count_(0), is_bf16_(use_bf16) {

        if (is_bf16_) {
            data_size_ = dim_ * sizeof(uint16_t) + sizeof(labeltype);
        } else {
            data_size_ = dim_ * sizeof(float) + sizeof(labeltype);
        }
        data_ = (char*)malloc(maxelements_ * data_size_);
        if (!data_) {
            throw std::runtime_error("Failed to allocate memory for BruteforceBatchSearch");
        }
    }

    ~BruteforceBatchSearch() {
        if (data_) {
            free(data_);
        }
    }

    void addPoint(const void* data, labeltype label) {
        if (cur_element_count_ >= maxelements_) {
            throw std::runtime_error("BruteforceBatchSearch: max elements reached");
        }

        char* dest = data_ + cur_element_count_ * data_size_;
        if (is_bf16_) {
            memcpy(dest, data, dim_ * sizeof(uint16_t));
            memcpy(dest + dim_ * sizeof(uint16_t), &label, sizeof(labeltype));
        } else {
            memcpy(dest, data, dim_ * sizeof(float));
            memcpy(dest + dim_ * sizeof(float), &label, sizeof(labeltype));
        }
        cur_element_count_++;
    }

    labeltype getLabel(size_t idx) const {
        if (is_bf16_) {
            return *(labeltype*)(data_ + idx * data_size_ + dim_ * sizeof(uint16_t));
        } else {
            return *(labeltype*)(data_ + idx * data_size_ + dim_ * sizeof(float));
        }
    }

    const void* getData(size_t idx) const {
        return data_ + idx * data_size_;
    }

    // Batch search: process multiple queries at once
    // Returns vector of priority queues, one per query
    // Uses AMX batch operations when available
    std::vector<std::priority_queue<std::pair<dist_t, labeltype>>>
    searchKnnBatch(const void* queries, size_t num_queries, size_t k, int num_threads = 0) {

        if (num_threads > 0) {
            omp_set_num_threads(num_threads);
        }

        std::vector<std::priority_queue<std::pair<dist_t, labeltype>>> results(num_queries);

        const size_t QUERY_BATCH = 16;
        const size_t DB_BATCH = 16;

        // Allocate temporary storage for batch inner products
        std::vector<float> batch_dists(QUERY_BATCH * DB_BATCH);

        // Process queries in batches
        for (size_t q_start = 0; q_start < num_queries; q_start += QUERY_BATCH) {
            size_t q_end = std::min(q_start + QUERY_BATCH, num_queries);
            size_t q_batch_size = q_end - q_start;

            // Initialize result queues for this batch
            for (size_t q = q_start; q < q_end; q++) {
                results[q] = std::priority_queue<std::pair<dist_t, labeltype>>();
            }

            // Process DB in batches
            for (size_t db_start = 0; db_start < cur_element_count_; db_start += DB_BATCH) {
                size_t db_end = std::min(db_start + DB_BATCH, cur_element_count_);
                size_t db_batch_size = db_end - db_start;

                // Compute batch inner products
                computeBatchDistances(queries, q_start, q_batch_size,
                                     db_start, db_batch_size,
                                     batch_dists.data());

                // Update top-k for each query in batch
                for (size_t q = 0; q < q_batch_size; q++) {
                    auto& pq = results[q_start + q];
                    for (size_t d = 0; d < db_batch_size; d++) {
                        float dist = batch_dists[q * db_batch_size + d];
                        // For inner product, we want max, so use negative distance
                        dist_t neg_dist = -dist;
                        labeltype label = getLabel(db_start + d);

                        if (pq.size() < k) {
                            pq.emplace(neg_dist, label);
                        } else if (neg_dist < pq.top().first) {
                            pq.pop();
                            pq.emplace(neg_dist, label);
                        }
                    }
                }
            }
        }

        return results;
    }

    // Parallel batch search using OpenMP
    std::vector<std::priority_queue<std::pair<dist_t, labeltype>>>
    searchKnnBatchParallel(const void* queries, size_t num_queries, size_t k, int num_threads = 0) {

        if (num_threads > 0) {
            omp_set_num_threads(num_threads);
        }

        std::vector<std::priority_queue<std::pair<dist_t, labeltype>>> results(num_queries);

        // Each thread processes a subset of queries
        #pragma omp parallel
        {
            const size_t DB_BATCH = 256;  // Larger batch for better cache usage
            std::vector<float> local_dists(DB_BATCH);

            #pragma omp for schedule(dynamic, 1)
            for (size_t q = 0; q < num_queries; q++) {
                auto& pq = results[q];

                // Process all DB vectors for this query
                for (size_t db_start = 0; db_start < cur_element_count_; db_start += DB_BATCH) {
                    size_t db_end = std::min(db_start + DB_BATCH, cur_element_count_);
                    size_t db_batch_size = db_end - db_start;

                    // Compute distances for this batch
                    computeSingleQueryBatchDistances(queries, q, db_start, db_batch_size, local_dists.data());

                    // Update top-k
                    for (size_t d = 0; d < db_batch_size; d++) {
                        dist_t neg_dist = -local_dists[d];
                        labeltype label = getLabel(db_start + d);

                        if (pq.size() < k) {
                            pq.emplace(neg_dist, label);
                        } else if (neg_dist < pq.top().first) {
                            pq.pop();
                            pq.emplace(neg_dist, label);
                        }
                    }
                }
            }
        }

        return results;
    }

private:
    // Compute batch distances: q_batch_size queries x db_batch_size DB vectors
    // 방향 1+3: thread_local 버퍼 사용, 복사 최소화
    void computeBatchDistances(const void* queries, size_t q_start, size_t q_batch_size,
                               size_t db_start, size_t db_batch_size,
                               float* results) {
        BruteforceBatchBuffers& buf = getBFBuffers();

        if (is_bf16_) {
            const uint16_t* query_data = static_cast<const uint16_t*>(queries) + q_start * dim_;

#if defined(USE_AMX)
            // 방향 3: 포인터 배열로 직접 접근 (16개 이하일 때)
            if (q_batch_size <= 16 && db_batch_size <= 16) {
                // DB 포인터 수집 (복사 없음)
                for (size_t d = 0; d < db_batch_size; d++) {
                    buf.db_ptrs_bf16[d] = static_cast<const uint16_t*>(getData(db_start + d));
                }
                // 패딩용 빈 포인터
                for (size_t d = db_batch_size; d < 16; d++) {
                    buf.db_ptrs_bf16[d] = buf.db_ptrs_bf16[0];  // 더미
                }

                InnerProductBatchAMX16x16Direct(query_data, buf.db_ptrs_bf16, buf.batch_dists, dim_);

                // 결과 복사
                for (size_t q = 0; q < q_batch_size; q++) {
                    for (size_t d = 0; d < db_batch_size; d++) {
                        results[q * db_batch_size + d] = buf.batch_dists[q * 16 + d];
                    }
                }
            } else {
                // 큰 배치: 기존 방식 (thread_local 버퍼 사용)
                for (size_t d = 0; d < db_batch_size; d++) {
                    const uint16_t* db_ptr = static_cast<const uint16_t*>(getData(db_start + d));
                    memcpy(&buf.db_batch_bf16[d * dim_], db_ptr, dim_ * sizeof(uint16_t));
                }
                InnerProductBatchAMX(query_data, buf.db_batch_bf16, results,
                                    q_batch_size, db_batch_size, dim_);
            }
#else
            // Non-AMX: thread_local 버퍼 사용
            for (size_t d = 0; d < db_batch_size; d++) {
                const uint16_t* db_ptr = static_cast<const uint16_t*>(getData(db_start + d));
                memcpy(&buf.db_batch_bf16[d * dim_], db_ptr, dim_ * sizeof(uint16_t));
            }
            InnerProductBatchScalar(query_data, buf.db_batch_bf16, results,
                                   q_batch_size, db_batch_size, dim_);
#endif
        } else {
            const float* query_data = static_cast<const float*>(queries) + q_start * dim_;

            // FP32: thread_local 버퍼 사용
            for (size_t d = 0; d < db_batch_size; d++) {
                const float* db_ptr = static_cast<const float*>(getData(db_start + d));
                memcpy(&buf.db_batch_fp32[d * dim_], db_ptr, dim_ * sizeof(float));
            }

#if defined(__AVX512F__)
            InnerProductBatchAVX512FP32(query_data, buf.db_batch_fp32, results,
                                        q_batch_size, db_batch_size, dim_);
#elif defined(__AVX__)
            InnerProductBatchAVX2FP32(query_data, buf.db_batch_fp32, results,
                                      q_batch_size, db_batch_size, dim_);
#else
            InnerProductBatchScalarFP32(query_data, buf.db_batch_fp32, results,
                                        q_batch_size, db_batch_size, dim_);
#endif
        }
    }

    // Compute distances for single query against batch of DB vectors
    // 이 함수는 복사가 필요 없음 - 직접 접근
    void computeSingleQueryBatchDistances(const void* queries, size_t q_idx,
                                          size_t db_start, size_t db_batch_size,
                                          float* results) {
        if (is_bf16_) {
            const uint16_t* query_data = static_cast<const uint16_t*>(queries) + q_idx * dim_;

            auto bf16_to_float = [](uint16_t bf16) -> float {
                uint32_t tmp = static_cast<uint32_t>(bf16) << 16;
                return *reinterpret_cast<float*>(&tmp);
            };

#if defined(USE_AMX)
            // AMX로 16개씩 처리
            BruteforceBatchBuffers& buf = getBFBuffers();
            size_t d = 0;
            for (; d + 16 <= db_batch_size; d += 16) {
                for (size_t i = 0; i < 16; i++) {
                    buf.db_ptrs_bf16[i] = static_cast<const uint16_t*>(getData(db_start + d + i));
                }
                // 쿼리 1개를 16개로 복제해서 처리 (비효율적이지만 AMX 사용)
                // 실제로는 single query에서는 AMX 대신 scalar가 더 효율적
                for (size_t i = 0; i < 16; i++) {
                    const uint16_t* db_ptr = buf.db_ptrs_bf16[i];
                    float sum = 0.0f;
                    for (size_t j = 0; j < dim_; j++) {
                        sum += bf16_to_float(query_data[j]) * bf16_to_float(db_ptr[j]);
                    }
                    results[d + i] = sum;
                }
            }
            // 나머지
            for (; d < db_batch_size; d++) {
                const uint16_t* db_ptr = static_cast<const uint16_t*>(getData(db_start + d));
                float sum = 0.0f;
                for (size_t i = 0; i < dim_; i++) {
                    sum += bf16_to_float(query_data[i]) * bf16_to_float(db_ptr[i]);
                }
                results[d] = sum;
            }
#else
            for (size_t d = 0; d < db_batch_size; d++) {
                const uint16_t* db_ptr = static_cast<const uint16_t*>(getData(db_start + d));
                float sum = 0.0f;
                for (size_t i = 0; i < dim_; i++) {
                    sum += bf16_to_float(query_data[i]) * bf16_to_float(db_ptr[i]);
                }
                results[d] = sum;
            }
#endif
        } else {
            const float* query_data = static_cast<const float*>(queries) + q_idx * dim_;

            for (size_t d = 0; d < db_batch_size; d++) {
                const float* db_ptr = static_cast<const float*>(getData(db_start + d));
                float sum = 0.0f;

#if defined(__AVX512F__)
                __m512 vsum = _mm512_setzero_ps();
                size_t i = 0;
                for (; i + 16 <= dim_; i += 16) {
                    __m512 q_vec = _mm512_loadu_ps(query_data + i);
                    __m512 d_vec = _mm512_loadu_ps(db_ptr + i);
                    vsum = _mm512_fmadd_ps(q_vec, d_vec, vsum);
                }
                sum = _mm512_reduce_add_ps(vsum);
                for (; i < dim_; i++) {
                    sum += query_data[i] * db_ptr[i];
                }
#elif defined(__AVX__)
                __m256 vsum = _mm256_setzero_ps();
                size_t i = 0;
                for (; i + 8 <= dim_; i += 8) {
                    __m256 q_vec = _mm256_loadu_ps(query_data + i);
                    __m256 d_vec = _mm256_loadu_ps(db_ptr + i);
                    vsum = _mm256_fmadd_ps(q_vec, d_vec, vsum);
                }
                // Horizontal sum
                __m128 hi = _mm256_extractf128_ps(vsum, 1);
                __m128 lo = _mm256_castps256_ps128(vsum);
                __m128 sum128 = _mm_add_ps(hi, lo);
                sum128 = _mm_hadd_ps(sum128, sum128);
                sum128 = _mm_hadd_ps(sum128, sum128);
                sum = _mm_cvtss_f32(sum128);
                for (; i < dim_; i++) {
                    sum += query_data[i] * db_ptr[i];
                }
#else
                for (size_t i = 0; i < dim_; i++) {
                    sum += query_data[i] * db_ptr[i];
                }
#endif
                results[d] = sum;
            }
        }
    }
};

}  // namespace hnswlib
