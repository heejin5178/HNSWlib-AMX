#include <iostream>
#include <fstream>
#include <queue>
#include <chrono>
#include <map>
#include <atomic>
#include <cstdlib>
#include <omp.h>
#include "../../hnswlib/hnswlib.h"
#include "../../hnswlib/bruteforce_batch.h"

#include <unordered_set>

using namespace std;
using namespace hnswlib;

// FP32 to BF16 conversion helper
inline uint16_t float_to_bf16(float val) {
    return (*reinterpret_cast<uint32_t*>(&val)) >> 16;
}

// Config parser for key=value format
map<string, string> parse_config(const string &config_path) {
    map<string, string> config;
    ifstream file(config_path);
    if (!file.is_open()) {
        cerr << "Warning: Could not open config file: " << config_path << endl;
        return config;
    }
    string line;
    while (getline(file, line)) {
        if (line.empty() || line[0] == '#') continue;
        size_t pos = line.find('=');
        if (pos != string::npos) {
            string key = line.substr(0, pos);
            string value = line.substr(pos + 1);
            key.erase(0, key.find_first_not_of(" \t"));
            key.erase(key.find_last_not_of(" \t") + 1);
            value.erase(0, value.find_first_not_of(" \t"));
            value.erase(value.find_last_not_of(" \t") + 1);
            config[key] = value;
        }
    }
    file.close();
    return config;
}

int get_config_int(const map<string, string> &config, const string &key, int default_value) {
    auto it = config.find(key);
    if (it != config.end()) {
        return stoi(it->second);
    }
    return default_value;
}

double get_config_double(const map<string, string> &config, const string &key, double default_value) {
    auto it = config.find(key);
    if (it != config.end()) {
        return stod(it->second);
    }
    return default_value;
}

string get_config_string(const map<string, string> &config, const string &key, const string &default_value) {
    auto it = config.find(key);
    if (it != config.end()) {
        return it->second;
    }
    return default_value;
}

class StopW {
    std::chrono::steady_clock::time_point time_begin;
public:
    StopW() {
        time_begin = std::chrono::steady_clock::now();
    }

    float getElapsedTimeMicro() {
        std::chrono::steady_clock::time_point time_end = std::chrono::steady_clock::now();
        return (std::chrono::duration_cast<std::chrono::microseconds>(time_end - time_begin).count());
    }

    void reset() {
        time_begin = std::chrono::steady_clock::now();
    }
};

#if defined(_WIN32)
#include <windows.h>
#include <psapi.h>
#elif defined(__unix__) || defined(__unix) || defined(unix) || (defined(__APPLE__) && defined(__MACH__))
#include <unistd.h>
#include <sys/resource.h>
#if defined(__APPLE__) && defined(__MACH__)
#include <mach/mach.h>
#elif defined(__linux__) || defined(__linux) || defined(linux) || defined(__gnu_linux__)
#endif
#endif

static size_t getCurrentRSS() {
#if defined(_WIN32)
    PROCESS_MEMORY_COUNTERS info;
    GetProcessMemoryInfo(GetCurrentProcess(), &info, sizeof(info));
    return (size_t)info.WorkingSetSize;
#elif defined(__linux__) || defined(__linux) || defined(linux) || defined(__gnu_linux__)
    long rss = 0L;
    FILE *fp = NULL;
    if ((fp = fopen("/proc/self/statm", "r")) == NULL)
        return (size_t) 0L;
    if (fscanf(fp, "%*s%ld", &rss) != 1) {
        fclose(fp);
        return (size_t) 0L;
    }
    fclose(fp);
    return (size_t) rss * (size_t) sysconf(_SC_PAGESIZE);
#else
    return (size_t)0L;
#endif
}

// QPS benchmark with batch search
template<typename T>
static double
test_qps_batch(
    T *massQ,
    size_t qsize,
    BruteforceBatchSearch<float> &appr_alg,
    size_t vecdim,
    size_t k,
    double duration_seconds,
    int num_threads,
    size_t &total_queries_out,
    size_t query_batch_size = 16) {

    if (num_threads > 0) {
        omp_set_num_threads(num_threads);
    }

    std::atomic<bool> stop_flag(false);
    std::atomic<size_t> total_queries(0);

    auto start_time = std::chrono::steady_clock::now();

    // Use parallel batch search
    #pragma omp parallel
    {
        size_t local_queries = 0;

        while (!stop_flag.load(std::memory_order_relaxed)) {
            // Each thread processes a batch of queries
            size_t batch_start = (local_queries * query_batch_size) % qsize;
            size_t actual_batch = std::min(query_batch_size, qsize - batch_start);

            // Use batch search
            auto results = appr_alg.searchKnnBatch(
                massQ + batch_start * vecdim,
                actual_batch, k, 1);  // Use 1 thread inside since we're already parallel

            local_queries += actual_batch;

            // Check time periodically
            if ((local_queries / query_batch_size) % 10 == 0) {
                auto now = std::chrono::steady_clock::now();
                double elapsed = std::chrono::duration<double>(now - start_time).count();
                if (elapsed >= duration_seconds) {
                    stop_flag.store(true, std::memory_order_relaxed);
                }
            }
        }

        total_queries.fetch_add(local_queries, std::memory_order_relaxed);
    }

    auto end_time = std::chrono::steady_clock::now();
    double actual_duration = std::chrono::duration<double>(end_time - start_time).count();

    total_queries_out = total_queries.load();
    return static_cast<double>(total_queries_out) / actual_duration;
}

// QPS benchmark with true batch search (uses AMX 16x16 tiles)
template<typename T>
static double
test_qps_batch_v2(
    T *massQ,
    size_t qsize,
    BruteforceBatchSearch<float> &appr_alg,
    size_t vecdim,
    size_t k,
    double duration_seconds,
    int num_threads,
    size_t &total_queries_out,
    size_t query_batch_size = 16) {  // 16으로 변경 (AMX 타일 크기)

    if (num_threads > 0) {
        omp_set_num_threads(num_threads);
    }

    std::atomic<size_t> total_queries(0);
    std::atomic<bool> stop_flag(false);
    auto start_time = std::chrono::steady_clock::now();

    // 각 스레드가 독립적으로 배치 처리
    #pragma omp parallel
    {
        size_t local_queries = 0;
        size_t local_batch_count = 0;
        size_t local_offset = omp_get_thread_num() * query_batch_size;  // 스레드별 시작점

        while (!stop_flag.load(std::memory_order_relaxed)) {
            // 현재 배치 시작 위치 계산
            size_t q_start = (local_offset + local_queries) % qsize;
            size_t batch_size = std::min(query_batch_size, qsize - q_start);

            if (batch_size == 0) batch_size = query_batch_size;  // 안전장치

            // 진정한 배치 검색 (AMX 16x16 타일 사용)
            auto results = appr_alg.searchKnnBatch(
                massQ + q_start * vecdim,
                batch_size, k, 1);  // 내부에서 스레드 1개 사용 (이미 병렬)

            local_queries += batch_size;
            local_batch_count++;

            // 매 100 배치마다 시간 체크 (batch_size와 무관하게)
            if (local_batch_count % 100 == 0) {
                auto now = std::chrono::steady_clock::now();
                double elapsed = std::chrono::duration<double>(now - start_time).count();
                if (elapsed >= duration_seconds) {
                    stop_flag.store(true, std::memory_order_relaxed);
                }
            }
        }

        total_queries.fetch_add(local_queries, std::memory_order_relaxed);
    }

    auto end_time = std::chrono::steady_clock::now();
    double actual_duration = std::chrono::duration<double>(end_time - start_time).count();

    total_queries_out = total_queries.load();

    std::cout << "[DEBUG] Requested duration: " << duration_seconds
              << "s, Actual duration: " << actual_duration << "s" << std::endl;

    return static_cast<double>(total_queries_out) / actual_duration;
}

template<typename T>
static void
test_qps_benchmark_batch(
    T *massQ,
    size_t qsize,
    BruteforceBatchSearch<float> &appr_alg,
    size_t vecdim,
    size_t k,
    ostream &log,
    double duration_seconds,
    int num_threads = 0,
    size_t query_batch_size = 16) {  // AMX 타일 크기에 맞춤

    int actual_threads = num_threads;
    if (actual_threads <= 0) {
        actual_threads = omp_get_max_threads();
    }
    log << "QPS Batch Benchmark: Running for " << duration_seconds << " seconds with "
        << actual_threads << " threads, query_batch_size=" << query_batch_size << "\n";

    size_t total_queries = 0;

    // Use version 2 (parallel batch search)
    double qps = test_qps_batch_v2(massQ, qsize, appr_alg, vecdim, k,
                                   duration_seconds, num_threads, total_queries, query_batch_size);

#if defined(USE_AMX_BF16)
    log << "BatchSearch (BF16, AMX Batch)\tQPS: " << qps << "\tTotal queries: " << total_queries << "\n";
#elif defined(USE_AMX)
    log << "BatchSearch (FP32, AMX Batch)\tQPS: " << qps << "\tTotal queries: " << total_queries << "\n";
#else
    log << "BatchSearch (FP32)\tQPS: " << qps << "\tTotal queries: " << total_queries << "\n";
#endif
}

void sift_test1B_batch(const string &config_path = "", const string &log_path = "") {
    // Setup log stream
    ofstream log_file;
    ostream *log_stream = &cout;
    if (!log_path.empty()) {
        log_file.open(log_path);
        if (log_file.is_open()) {
            log_stream = &log_file;
            cout << "Logging to: " << log_path << endl;
        } else {
            cerr << "Warning: Could not open log file: " << log_path << ", using stdout" << endl;
        }
    }
    ostream &log = *log_stream;

    // Parse config
    map<string, string> config;
    if (!config_path.empty()) {
        config = parse_config(config_path);
        log << "Loaded config from: " << config_path << endl;
    }

    // Read config values
    int subset_size_milllions = get_config_int(config, "subset_size_millions", 10);
    size_t qsize = get_config_int(config, "qsize", 1000);
    size_t vecdim = get_config_int(config, "vecdim", 128);
    int num_threads = get_config_int(config, "num_threads", 0);
    double duration_seconds = get_config_double(config, "duration_seconds", 0.0);
    size_t query_batch_size = get_config_int(config, "query_batch_size", 64);

    // Override with environment variables
    const char* env_duration = getenv("DURATION_SECONDS");
    if (env_duration != nullptr) {
        duration_seconds = stod(env_duration);
    }
    const char* env_batch = getenv("QUERY_BATCH_SIZE");
    if (env_batch != nullptr) {
        query_batch_size = stoul(env_batch);
    }

    log << "Config: subset_size_millions=" << subset_size_milllions
        << ", qsize=" << qsize
        << ", vecdim=" << vecdim
        << ", num_threads=" << (num_threads > 0 ? num_threads : omp_get_max_threads())
        << ", duration_seconds=" << duration_seconds
        << ", query_batch_size=" << query_batch_size << endl;

    size_t vecsize = subset_size_milllions * 1000000;

    string path_q = get_config_string(config, "path_query", "/mnt/ceph/heejin/bigann/bigann_query.bvecs");
    string path_data = get_config_string(config, "path_data", "/mnt/ceph/heejin/bigann/bigann_base.bvecs");

    unsigned char *massb = new unsigned char[vecdim];

    // Load queries
    log << "Loading queries:\n";
#if defined(USE_AMX_BF16)
    uint16_t *massQ = new uint16_t[qsize * vecdim];
    bool use_bf16 = true;
#else
    float *massQ = new float[qsize * vecdim];
    bool use_bf16 = false;
#endif

    ifstream inputQ(path_q, ios::binary);
    for (size_t i = 0; i < qsize; i++) {
        int in = 0;
        inputQ.read((char *) &in, 4);
        if (in != (int)vecdim) {
            log << "file error: expected vecdim=" << vecdim << ", got " << in << endl;
            exit(1);
        }
        inputQ.read((char *) massb, in);
        for (size_t j = 0; j < vecdim; j++) {
#if defined(USE_AMX_BF16)
            float tmp = (float) massb[j];
            massQ[i * vecdim + j] = float_to_bf16(tmp);
#else
            massQ[i * vecdim + j] = (float) massb[j];
#endif
        }
    }
    inputQ.close();
    log << "Loaded " << qsize << " queries\n";

    // Build index using BruteforceBatchSearch
#if defined(USE_AMX_BF16)
    uint16_t *mass = new uint16_t[vecdim];
    log << "Using BruteforceBatchSearch with BF16 storage\n";
#else
    float *mass = new float[vecdim];
    log << "Using BruteforceBatchSearch with FP32 storage\n";
#endif

    BruteforceBatchSearch<float> *appr_alg = new BruteforceBatchSearch<float>(vecdim, vecsize, use_bf16);

    ifstream input(path_data, ios::binary);
    int in = 0;

    log << "Building BruteforceBatchSearch index:\n";
    StopW stopw = StopW();
    StopW stopw_full = StopW();
    size_t report_every = 100000;

    for (size_t i = 0; i < vecsize; i++) {
        input.read((char *) &in, 4);
        if (in != (int)vecdim) {
            log << "file error: expected vecdim=" << vecdim << ", got " << in << endl;
            exit(1);
        }
        input.read((char *) massb, in);
        for (size_t j = 0; j < vecdim; j++) {
#if defined(USE_AMX_BF16)
            float tmp = (float) massb[j];
            mass[j] = float_to_bf16(tmp);
#else
            mass[j] = (float) massb[j];
#endif
        }

        appr_alg->addPoint((void *) mass, (size_t) i);

        if ((i + 1) % report_every == 0) {
            log << (i + 1) / (0.01 * vecsize) << " %, "
                << report_every / (1000.0 * 1e-6 * stopw.getElapsedTimeMicro()) << " kips " << " Mem: "
                << getCurrentRSS() / 1000000 << " Mb \n";
            stopw.reset();
        }
    }
    input.close();
    log << "Build time:" << 1e-6 * stopw_full.getElapsedTimeMicro() << "  seconds\n";

    size_t k = 1;

    if (duration_seconds > 0) {
        // QPS benchmark mode
        log << "Running QPS batch benchmark mode\n";
        test_qps_benchmark_batch(massQ, qsize, *appr_alg, vecdim, k, log,
                                 duration_seconds, num_threads, query_batch_size);
    } else {
        log << "Please set duration_seconds > 0 for batch QPS benchmark\n";
    }

    log << "Actual memory usage: " << getCurrentRSS() / 1000000 << " Mb \n";

    delete[] massQ;
    delete[] mass;
    delete[] massb;
    delete appr_alg;

    if (log_file.is_open()) {
        log_file.close();
    }
}
