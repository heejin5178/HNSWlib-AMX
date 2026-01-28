#include <iostream>
#include <fstream>
#include <queue>
#include <chrono>
#include <map>
#include <atomic>
#include <cstdlib>
#include <omp.h>
#include "../../hnswlib/hnswlib.h"
#include<utility>

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
        // Skip empty lines and comments
        if (line.empty() || line[0] == '#') continue;

        size_t pos = line.find('=');
        if (pos != string::npos) {
            string key = line.substr(0, pos);
            string value = line.substr(pos + 1);
            // Trim whitespace
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



/*
* Author:  David Robert Nadeau
* Site:    http://NadeauSoftware.com/
* License: Creative Commons Attribution 3.0 Unported License
*          http://creativecommons.org/licenses/by/3.0/deed.en_US
*/

#if defined(_WIN32)
#include <windows.h>
#include <psapi.h>

#elif defined(__unix__) || defined(__unix) || defined(unix) || (defined(__APPLE__) && defined(__MACH__))

#include <unistd.h>
#include <sys/resource.h>

#if defined(__APPLE__) && defined(__MACH__)
#include <mach/mach.h>

#elif (defined(_AIX) || defined(__TOS__AIX__)) || (defined(__sun__) || defined(__sun) || defined(sun) && (defined(__SVR4) || defined(__svr4__)))
#include <fcntl.h>
#include <procfs.h>

#elif defined(__linux__) || defined(__linux) || defined(linux) || defined(__gnu_linux__)

#endif

#else
#error "Cannot define getPeakRSS( ) or getCurrentRSS( ) for an unknown OS."
#endif


/**
* Returns the peak (maximum so far) resident set size (physical
* memory use) measured in bytes, or zero if the value cannot be
* determined on this OS.
*/
static size_t getPeakRSS() {
#if defined(_WIN32)
    /* Windows -------------------------------------------------- */
    PROCESS_MEMORY_COUNTERS info;
    GetProcessMemoryInfo(GetCurrentProcess(), &info, sizeof(info));
    return (size_t)info.PeakWorkingSetSize;

#elif (defined(_AIX) || defined(__TOS__AIX__)) || (defined(__sun__) || defined(__sun) || defined(sun) && (defined(__SVR4) || defined(__svr4__)))
    /* AIX and Solaris ------------------------------------------ */
    struct psinfo psinfo;
    int fd = -1;
    if ((fd = open("/proc/self/psinfo", O_RDONLY)) == -1)
        return (size_t)0L;      /* Can't open? */
    if (read(fd, &psinfo, sizeof(psinfo)) != sizeof(psinfo)) {
        close(fd);
        return (size_t)0L;      /* Can't read? */
    }
    close(fd);
    return (size_t)(psinfo.pr_rssize * 1024L);

#elif defined(__unix__) || defined(__unix) || defined(unix) || (defined(__APPLE__) && defined(__MACH__))
    /* BSD, Linux, and OSX -------------------------------------- */
    struct rusage rusage;
    getrusage(RUSAGE_SELF, &rusage);
#if defined(__APPLE__) && defined(__MACH__)
    return (size_t)rusage.ru_maxrss;
#else
    return (size_t) (rusage.ru_maxrss * 1024L);
#endif

#else
    /* Unknown OS ----------------------------------------------- */
    return (size_t)0L;          /* Unsupported. */
#endif
}


/**
* Returns the current resident set size (physical memory use) measured
* in bytes, or zero if the value cannot be determined on this OS.
*/
static size_t getCurrentRSS() {
#if defined(_WIN32)
    /* Windows -------------------------------------------------- */
    PROCESS_MEMORY_COUNTERS info;
    GetProcessMemoryInfo(GetCurrentProcess(), &info, sizeof(info));
    return (size_t)info.WorkingSetSize;

#elif defined(__APPLE__) && defined(__MACH__)
    /* OSX ------------------------------------------------------ */
    struct mach_task_basic_info info;
    mach_msg_type_number_t infoCount = MACH_TASK_BASIC_INFO_COUNT;
    if (task_info(mach_task_self(), MACH_TASK_BASIC_INFO,
        (task_info_t)&info, &infoCount) != KERN_SUCCESS)
        return (size_t)0L;      /* Can't access? */
    return (size_t)info.resident_size;

#elif defined(__linux__) || defined(__linux) || defined(linux) || defined(__gnu_linux__)
    /* Linux ---------------------------------------------------- */
    long rss = 0L;
    FILE *fp = NULL;
    if ((fp = fopen("/proc/self/statm", "r")) == NULL)
        return (size_t) 0L;      /* Can't open? */
    if (fscanf(fp, "%*s%ld", &rss) != 1) {
        fclose(fp);
        return (size_t) 0L;      /* Can't read? */
    }
    fclose(fp);
    return (size_t) rss * (size_t) sysconf(_SC_PAGESIZE);

#else
    /* AIX, BSD, Solaris, and Unknown OS ------------------------ */
    return (size_t)0L;          /* Unsupported. */
#endif
}


static void
get_gt(
    unsigned int *massQA,
    size_t qsize,
    vector<std::priority_queue<std::pair<float, labeltype>>> &answers,
    size_t k,
    ostream &log) {
    log << "get_gt in\n";
    (vector<std::priority_queue<std::pair<float, labeltype >>>(qsize)).swap(answers);
    log << "qsize: " << qsize << "\n";
    for (int i = 0; i < qsize; i++) {
        for (int j = 0; j < k; j++) {
            answers[i].emplace(0.0f, massQA[1000 * i + j]);
        }
    }
}

// static float
// test_approx(
//     unsigned char *massQ,
//     size_t vecsize,
//     size_t qsize,
//     HierarchicalNSW<int> &appr_alg,
//     size_t vecdim,
//     vector<std::priority_queue<std::pair<int, labeltype>>> &answers,
//     size_t k) {
//     size_t correct = 0;
//     size_t total = 0;
//     // uncomment to test in parallel mode:
//     //#pragma omp parallel for
//     for (int i = 0; i < qsize; i++) {
//         std::priority_queue<std::pair<int, labeltype >> result = appr_alg.searchKnn(massQ + vecdim * i, k);
//         std::priority_queue<std::pair<int, labeltype >> gt(answers[i]);
//         unordered_set<labeltype> g;
//         total += gt.size();
//
//         while (gt.size()) {
//             g.insert(gt.top().second);
//             gt.pop();
//         }
//
//         while (result.size()) {
//             if (g.find(result.top().second) != g.end()) {
//                 correct++;
//             } else {
//             }
//             result.pop();
//         }
//     }
//     return 1.0f * correct / total;
// }

template<typename T>
static float
test_approx(
    T *massQ,
    size_t vecsize,
    size_t qsize,
    HierarchicalNSW<float> &appr_alg,
    size_t vecdim,
    vector<std::priority_queue<std::pair<float, labeltype>>> &answers,
    size_t k,
    int num_threads = 0) {
    size_t correct = 0;
    size_t total = 0;

    if (num_threads > 0) {
        omp_set_num_threads(num_threads);
    }

    #pragma omp parallel for reduction(+:correct, total)
    for (int i = 0; i < (int)qsize; i++) {
        std::priority_queue<std::pair<float, labeltype >> result = appr_alg.searchKnn(massQ + vecdim * i, k);
        std::priority_queue<std::pair<float, labeltype >> gt(answers[i]);
        unordered_set<labeltype> g;

        size_t local_total = gt.size();
        total += local_total;

        while (gt.size()) {
            g.insert(gt.top().second);
            gt.pop();
        }

        size_t local_correct = 0;
        while (result.size()) {
            if (g.find(result.top().second) != g.end()) {
                local_correct++;
            }
            result.pop();
        }
        correct += local_correct;
    }
    return 1.0f * correct / total;
}

// QPS benchmark: run queries for fixed duration and measure throughput
template<typename T>
// static double
static std::pair<double, double> 
test_qps(
    T *massQ,
    size_t qsize,
    HierarchicalNSW<float> &appr_alg,
    size_t vecdim,
    size_t k,
    double duration_seconds,
    int num_threads,
    size_t &total_queries_out,
    vector<std::priority_queue<std::pair<float, labeltype>>> &answers) {

    if (num_threads > 0) {
        omp_set_num_threads(num_threads);
    }

    std::atomic<bool> stop_flag(false);
    std::atomic<size_t> total_queries(0);
    std::atomic<size_t> global_query_idx(0);
    std::atomic<size_t> correct(0);
    std::atomic<size_t> total(0);
    /*
    std::vector<size_t> index_checker;
    std::mutex mu_;
    */
    auto start_time = std::chrono::steady_clock::now();

    #pragma omp parallel
    {
        size_t local_queries = 0;
        int tid = omp_get_thread_num();

        while (!stop_flag.load(std::memory_order_acquire)) {
            // Use atomic counter to distribute queries evenly across all threads
            size_t i = global_query_idx.fetch_add(1, std::memory_order_relaxed) % qsize;

            std::priority_queue<std::pair<float, labeltype >> result = appr_alg.searchKnn(massQ + vecdim * i, k);
            std::priority_queue<std::pair<float, labeltype >> gt(answers[i]);
           
            unordered_set<labeltype> g;

            size_t local_total = gt.size();
            total += local_total;

            while (gt.size()) {
                g.insert(gt.top().second);
                gt.pop();
            }

            size_t local_correct = 0;
            while (result.size()) {
                if (g.find(result.top().second) != g.end()) {
                    local_correct++;
                }
                result.pop();
            }
            correct += local_correct;

            local_queries++;

            // Only master thread checks time periodically (every 1000 queries to reduce overhead)
            if (global_query_idx.load() % 1000 == 0) {
                auto now = std::chrono::steady_clock::now();
                double elapsed = std::chrono::duration<double>(now - start_time).count();
                if (elapsed >= duration_seconds) {
                    stop_flag.store(true, std::memory_order_release);
                }
            }
        }

        total_queries.fetch_add(local_queries, std::memory_order_relaxed);
    }

    /*
    std::sort(index_checker.begin(), index_checker.end());
    bool is_duplicate = std::adjacent_find(index_checker.begin(), index_checker.end()) != index_checker.end();
    if (is_duplicate) {
        fprintf(stderr, "dupliacte found !!\n");
        exit(1);
    }
    */
    auto end_time = std::chrono::steady_clock::now();
    double actual_duration = std::chrono::duration<double>(end_time - start_time).count();

    total_queries_out = total_queries.load();
    double recall = 1.0f * correct / total;
    return std::make_pair(static_cast<double>(total_queries_out) / actual_duration, recall);  // QPS
}

// static void
// test_vs_recall(
//     unsigned char *massQ,
//     size_t vecsize,
//     size_t qsize,
//     HierarchicalNSW<int> &appr_alg,
//     size_t vecdim,
//     vector<std::priority_queue<std::pair<int, labeltype>>> &answers,
//     size_t k) {
//     vector<size_t> efs;  // = { 10,10,10,10,10 };
//     for (int i = k; i < 30; i++) {
//         efs.push_back(i);
//     }
//     for (int i = 30; i < 100; i += 10) {
//         efs.push_back(i);
//     }
//     for (int i = 100; i < 500; i += 40) {
//         efs.push_back(i);
//     }
//     for (size_t ef : efs) {
//         appr_alg.setEf(ef);
//         StopW stopw = StopW();
//
//         float recall = test_approx(massQ, vecsize, qsize, appr_alg, vecdim, answers, k);
//         float time_us_per_query = stopw.getElapsedTimeMicro() / qsize;
//
//         cout << ef << "\t" << recall << "\t" << time_us_per_query << " us\n";
//         if (recall > 1.0) {
//             cout << recall << "\t" << time_us_per_query << " us\n";
//             break;
//         }
//     }
// }

template<typename T>
static void
test_vs_recall(
    T *massQ,
    size_t vecsize,
    size_t qsize,
    HierarchicalNSW<float> &appr_alg,
    size_t vecdim,
    vector<std::priority_queue<std::pair<float, labeltype>>> &answers,
    size_t k,
    ostream &log,
    int num_threads = 0) {

    int actual_threads = num_threads;
    if (actual_threads <= 0) {
        actual_threads = omp_get_max_threads();
    }
    log << "Running with " << actual_threads << " threads\n";

    StopW stopw = StopW();

    float recall = test_approx(massQ, vecsize, qsize, appr_alg, vecdim, answers, k, num_threads);
    float time_us_per_query = stopw.getElapsedTimeMicro() / qsize;
    float total_time_ms = stopw.getElapsedTimeMicro() / 1000.0f;

#if defined(USE_AMX_BF16)
    log << "HierarchicalNSW (Bf16InnerProductSpace, MT)\t" << recall << "\t" << time_us_per_query << " us/query\t" << total_time_ms << " ms total\n";
#else
    log << "HierarchicalNSW (InnerProductSpace float, MT)\t" << recall << "\t" << time_us_per_query << " us/query\t" << total_time_ms << " ms total\n";
#endif
}

// QPS benchmark mode: run for fixed duration
template<typename T>
static void
test_qps_benchmark(
    T *massQ,
    size_t qsize,
    HierarchicalNSW<float> &appr_alg,
    size_t vecdim,
    size_t k,
    ostream &log,
    double duration_seconds,
    vector<std::priority_queue<std::pair<float, labeltype>>> &answers,
    int num_threads = 0) {

    int actual_threads = num_threads;
    if (actual_threads <= 0) {
        actual_threads = omp_get_max_threads();
    }
    log << "QPS Benchmark: Running for " << duration_seconds << " seconds with " << actual_threads << " threads\n";
    size_t total_queries = 0;
    std::pair<double, double> result = test_qps(massQ, qsize, appr_alg, vecdim, k, duration_seconds, num_threads, total_queries, answers);
    double qps = result.first;
    double recall = result.second;

#if defined(USE_AMX_BF16)
    log << "HierarchicalNSW (Bf16InnerProductSpace, MT)\tQPS: " << qps << "\tTotal queries: " << total_queries << ", recall: " << recall  << "\n";
#else
    log << "HierarchicalNSW (InnerProductSpace float, MT)\tQPS: " << qps << "\tTotal queries: " << total_queries << ", recall: " << recall << "\n";
#endif
}

inline bool exists_test(const std::string &name) {
    ifstream f(name.c_str());
    return f.good();
}


void sift_test1B_mt(const string &config_path = "", const string &log_path = "") {
    // Setup log stream (file or cout)
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
    hnswlib::set_log_stream(&log);

    // Parse config file
    map<string, string> config;
    if (!config_path.empty()) {
        config = parse_config(config_path);
        log << "Loaded config from: " << config_path << endl;
    }

    // Read values from config or use defaults
    int subset_size_milllions = get_config_int(config, "subset_size_millions", 10);
    size_t qsize = get_config_int(config, "qsize", 1000);
    size_t vecdim = get_config_int(config, "vecdim", 128);
    int efConstruction = get_config_int(config, "efConstruction", 40);
    int M = get_config_int(config, "M", 16);
    int num_threads = get_config_int(config, "num_threads", 0);  // 0 means use default (all cores)
    double duration_seconds = get_config_double(config, "duration_seconds", 0.0);  // 0 means use fixed query count mode

    // Override with environment variables if set
    const char* env_duration = getenv("DURATION_SECONDS");
    if (env_duration != nullptr) {
        duration_seconds = stod(env_duration);
    }

    // Paths from config or defaults
    string path_q = get_config_string(config, "path_query", "/mnt/ceph/heejin/bigann/bigann_query.bvecs");
    string path_data = get_config_string(config, "path_data", "/mnt/ceph/heejin/bigann/bigann_base.bvecs");
    string path_gt_base = get_config_string(config, "path_gt_base", "/mnt/ceph/heejin/bigann/gnd/idx_%dM.ivecs");

    size_t vecsize = subset_size_milllions * 1000000;  // Use millions, not thousands
    // size_t vecsize = subset_size_milllions * 10000000;
    // size_t vecsize = subset_size_milllions * 1000;
    log << "Config: subset_size_millions=" << subset_size_milllions
        << ", qsize=" << qsize
        << ", vecdim=" << vecdim
        << ", num_threads=" << (num_threads > 0 ? num_threads : omp_get_max_threads())
        << ", duration_seconds=" << duration_seconds 
        << ", vecsize= " << vecsize << endl;


    char path_index[1024];
    char path_gt[1024];
    snprintf(path_index, sizeof(path_index), "/mnt/ceph/heejin/sift1b_%dm_ef_%d_M_%d.bin", subset_size_milllions, efConstruction, M);
    snprintf(path_gt, sizeof(path_gt), path_gt_base.c_str(), subset_size_milllions);

    unsigned char *massb = new unsigned char[vecdim];

    log << "Loading GT:\n";
    ifstream inputGT(path_gt, ios::binary);
    unsigned int *massQA = new unsigned int[qsize * 1000];
    for (int i = 0; i < qsize; i++) {
        int t;
        inputGT.read((char *) &t, 4);
        inputGT.read((char *) (massQA + 1000 * i), t * 4);
        if (t != 1000) {
            log << "err";
            return;
        }
    }
    inputGT.close();

    log << "Loading queries:\n";
#if defined(USE_AMX_BF16)
    uint16_t *massQ = new uint16_t[qsize * vecdim];  // BF16 queries
#else
    float *massQ = new float[qsize * vecdim];  // FP32 queries
#endif
    ifstream inputQ(path_q, ios::binary);

    for (int i = 0; i < qsize; i++) {
        int in = 0;
        inputQ.read((char *) &in, 4);
        if (in != vecdim) {
            log << "file error: expected vecdim=" << vecdim << ", got " << in << endl;
            exit(1);
        }
        inputQ.read((char *) massb, in);
        for (int j = 0; j < vecdim; j++) {
#if defined(USE_AMX_BF16)
            float tmp = (float) massb[j];
            massQ[i * vecdim + j] = float_to_bf16(tmp);  // Convert to BF16
#else
            massQ[i * vecdim + j] = (float) massb[j];  // Convert byte to float
#endif
        }
    }
    inputQ.close();


#if defined(USE_AMX_BF16)
    uint16_t *mass = new uint16_t[vecdim];  // BF16 buffer
    Bf16InnerProductSpace ipspace(vecdim);  // BF16 space
    log << "Using Bf16InnerProductSpace (BF16 storage)\n";
#else
    float *mass = new float[vecdim];  // FP32 buffer
    InnerProductSpace ipspace(vecdim);  // FP32 space
    log << "Using InnerProductSpace (FP32 storage)\n";
#endif
    ifstream input(path_data, ios::binary);
    int in = 0;

    HierarchicalNSW<float> *appr_alg;
    if (exists_test(path_index)) {
        log << "Loading index from " << path_index << ":\n";
        appr_alg = new HierarchicalNSW<float>(&ipspace, path_index, false);
        cout << "Actual memory usage: " << getCurrentRSS() / 1000000 << " Mb \n";
    } else {
        log << "Building index:\n";
        appr_alg = new HierarchicalNSW<float>(&ipspace, vecsize, M, efConstruction);

        input.read((char *) &in, 4);
        if (in != vecdim) {
            log << "file error: expected vecdim=" << vecdim << ", got " << in << endl;
            exit(1);
        }
        input.read((char *) massb, in);

        for (int j = 0; j < vecdim; j++) {
#if defined(USE_AMX_BF16)
            float tmp = (float) massb[j];
            mass[j] = float_to_bf16(tmp);  // Convert to BF16
#else
            mass[j] = (float) massb[j];  // Convert byte to float
#endif
        }
        appr_alg->addPoint((void *) (mass), (size_t) 0);
        int j1 = 0;
        StopW stopw = StopW();
        StopW stopw_full = StopW();
        size_t report_every = 100000;
#pragma omp parallel for
        for (int i = 1; i < vecsize; i++) {
#if defined(USE_AMX_BF16)
            uint16_t local_mass[vecdim];
#else
            float local_mass[vecdim];
#endif
            int j2 = 0;
#pragma omp critical
            {
                input.read((char *) &in, 4);
                if (in != vecdim) {
                    log << "file error: expected vecdim=" << vecdim << ", got " << in << endl;
                    exit(1);
                }
                input.read((char *) massb, in);

                for (int j = 0; j < vecdim; j++) {
#if defined(USE_AMX_BF16)
                    float tmp = (float) massb[j];
                    local_mass[j] = float_to_bf16(tmp);  // Convert to BF16
#else
                    local_mass[j] = (float) massb[j];  // Convert byte to float
#endif
                }
                j1++;
                j2 = j1;
                if (j1 % report_every == 0) {
                    cout << j1 / (0.01 * vecsize) << " %, "
                         << report_every / (1000.0 * 1e-6 * stopw.getElapsedTimeMicro()) << " kips " << " Mem: "
                         << getCurrentRSS() / 1000000 << " Mb \n";
                    stopw.reset();
                }
            }
            appr_alg->addPoint((void *) (local_mass), (size_t) j2);
        }
        input.close();
        cout << "Build time:" << 1e-6 * stopw_full.getElapsedTimeMicro() << "  seconds\n";
        appr_alg->saveIndex(path_index);
    }

    vector<std::priority_queue<std::pair<float, labeltype >>> answers;
    size_t k = 1;

    if (duration_seconds > 0) {
        // QPS benchmark mode: run for fixed duration
        log << "Running QPS benchmark mode\n";
        get_gt(massQA, qsize, answers, k, log);
        test_qps_benchmark(massQ, qsize, *appr_alg, vecdim, k, log, duration_seconds, answers, num_threads);
    } else {
        // Fixed query count mode: run all queries once and measure recall
        log << "Parsing gt:\n";
        get_gt(massQA, qsize, answers, k, log);
        log << "Loaded gt\n";
        for (int i = 0; i < 1; i++)
            test_vs_recall(massQ, vecsize, qsize, *appr_alg, vecdim, answers, k, log, num_threads);
    }
    log << "Actual memory usage: " << getCurrentRSS() / 1000000 << " Mb \n";

    if (log_file.is_open()) {
        log_file.close();
    }
    return;
}
