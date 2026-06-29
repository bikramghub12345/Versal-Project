/*
 * MBU_simulate.cc  –  Multi-Bit Upset (MBU) Fault Injection Simulator
 * ======================================================================
 *
 * CHANGES IN THIS VERSION:
 *   - FEATURE_MAPS renamed to INPUT_TENSOR throughout
 *   - Default bit counts:
 *       All targets except BUFFERS: 1 5 20 50 100 200 500 1000 2000 5000
 *                                   10000 20000 50000 100000
 *       BUFFERS only:               1 2 5 10 15 20 50 100 200 500 1000
 *   - Result CSVs packed into <target>/csv/ subfolder
 *   - Every CSV row labelled with injection_type="MBU" and target name
 *   - Every log line clearly shows [MBU][TARGET] for per-image tracking
 *   - Python plot script updated to read from csv/ subfolder
 *
 * FAULT INJECTION METHODS (per target):
 * ──────────────────────────────────────
 * INSTRUCTIONS : DDR4 /dev/mem @ (dpu_instr_addr_reg × 4096)  flip+restore
 *                Reg 0x50/0x54, HPC0 port, PFN encoding.
 *
 * WEIGHTS      : DDR4 /dev/mem @ dpu_base0_addr  flip+restore
 *                Reg 0x60/0x64, HP0 port, 1:1 physical, stable across runs.
 *
 * INPUT_TENSOR : CPU imgBuf XOR via inject_sbu() before execute_async().
 *                VART DMA copies corrupted pixels to DDR4 input region at
 *                DDR4_input_base + 2080 (VART header offset).
 *                True intermediate feature maps (REG_1 WORKSPACE) are
 *                DPU-internal and cannot be accessed between layers.
 *
 * BUFFERS      : DDR4 /dev/mem @ dpu_base3_addr  flip after wait()
 *                Reg 0x78/0x7C, HP0, 1:1 physical, changes each run.
 *                Read address fresh post-inference. No restore needed.
 *
 * DDR4 ADDRESS MAP (ResNet50, ZCU104):
 *   0x50  dpu_instr_addr  PFN x 4096 = CPU phys   742,492 B   HPC0
 *   0x60  dpu_base0_addr  1:1 HP0                 25,726,976 B
 *   0x68  dpu_base1_addr  1:1 HP0                  2,207,744 B (not injected)
 *   0x70  dpu_base2_addr  1:1 HP0                    152,608 B (image at +2080)
 *   0x78  dpu_base3_addr  1:1 HP0                      1,008 B (changes per run)
 *
 * BUILD:
 *   g++ -std=c++17 -O2 -o MBU_simulate src/MBU_simulate.cc \
 *       ../common/common.cpp \
 *       -I ./src -I ../common \
 *       -I /usr/include/opencv4 -I /usr/include/vitis_ai \
 *       $(pkg-config --cflags --libs opencv4) \
 *       -lvart-runner -lxir -lglog -lpthread
 *
 * RUN (must be root for /dev/mem access):
 *   ./MBU_simulate <model.xmodel> [target] [-v]
 *   target: instructions | weights | input_tensor | buffers | all
 */

#include <dirent.h>
#include <fcntl.h>
#include <stdio.h>
#include <stdlib.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <time.h>
#include <unistd.h>

#include <algorithm>
#include <cmath>
#include <cstdarg>
#include <cstring>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <map>
#include <memory>
#include <numeric>
#include <random>
#include <set>
#include <sstream>
#include <string>
#include <vector>

#include "common.h"
#include <opencv2/opencv.hpp>

using namespace std;
using namespace cv;

// ─────────────────────────────────────────────────────────────────────────────
// CONSTANTS
// ─────────────────────────────────────────────────────────────────────────────
#define TOP_K 5

// DDR4 region sizes (xir dump_bin + xir dump_reg, ResNet50 ZCU104)
static const size_t DDR4_INSTR_SIZE  = 742492;
static const size_t DDR4_WEIGHT_SIZE = 25726976;
static const size_t DDR4_OUTPUT_SIZE = 1008;
static const size_t DDR4_INPUT_HDR   = 2080;   // VART header before image pixels

// AXI control register base + offsets (DPUCZDX8G_1, from xclbinutil)
static const uint32_t DPU_CTRL_BASE  = 0x80000000;
static const uint32_t OFF_INSTR_LO   = 0x50;
static const uint32_t OFF_INSTR_HI   = 0x54;
static const uint32_t OFF_BASE0_LO   = 0x60;
static const uint32_t OFF_BASE0_HI   = 0x64;
static const uint32_t OFF_BASE3_LO   = 0x78;
static const uint32_t OFF_BASE3_HI   = 0x7C;

// Default bit counts per target (set at interactive prompt)
static const vector<int> DEFAULT_BITS_GENERAL = {
    1, 5, 20, 50, 100, 200, 500, 1000,
    2000, 5000, 10000, 20000, 50000, 100000
};
static const vector<int> DEFAULT_BITS_BUFFERS = {
    1, 2, 5, 10, 15, 20, 50, 100, 200, 500, 1000
};

static const string wordsPath = "./";

// ─────────────────────────────────────────────────────────────────────────────
// FAULT TARGET
// ─────────────────────────────────────────────────────────────────────────────
enum class FaultTarget {
    INSTRUCTIONS,
    WEIGHTS,
    INPUT_TENSOR,   // formerly FEATURE_MAPS — flips input imgBuf pixels
    BUFFERS,
    ALL
};

static string targetName(FaultTarget t) {
    switch(t) {
        case FaultTarget::INSTRUCTIONS: return "INSTRUCTIONS";
        case FaultTarget::WEIGHTS:      return "WEIGHTS";
        case FaultTarget::INPUT_TENSOR: return "INPUT_TENSOR";
        case FaultTarget::BUFFERS:      return "BUFFERS";
        case FaultTarget::ALL:          return "ALL";
    }
    return "UNKNOWN";
}
static string targetDirName(FaultTarget t) {
    switch(t) {
        case FaultTarget::INSTRUCTIONS: return "instructions";
        case FaultTarget::WEIGHTS:      return "weights";
        case FaultTarget::INPUT_TENSOR: return "input_tensor";
        case FaultTarget::BUFFERS:      return "buffers";
        case FaultTarget::ALL:          return "all";
    }
    return "unknown";
}

// ─────────────────────────────────────────────────────────────────────────────
// DDR4 DIRECT ACCESS STATE
// ─────────────────────────────────────────────────────────────────────────────
static int      g_devmem_fd    = -1;
static uint64_t g_instr_phys   = 0;   // (dpu_instr_addr_reg) << 12  (PFN decode)
static uint64_t g_weights_phys = 0;   // dpu_base0_addr, stable across runs

GraphInfo shapes;

// ─────────────────────────────────────────────────────────────────────────────
// LOGGING
// ─────────────────────────────────────────────────────────────────────────────
static FILE* g_logfp = nullptr;
static void sim_log(const char* fmt, ...) {
    va_list a1, a2;
    va_start(a1, fmt); vprintf(fmt, a1); va_end(a1);
    if (g_logfp) {
        va_start(a2, fmt); vfprintf(g_logfp, fmt, a2); va_end(a2);
        fflush(g_logfp);
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// CONTROL REGISTER READ
// ─────────────────────────────────────────────────────────────────────────────
static uint64_t read_ctrl_reg64(uint32_t off_lo, uint32_t off_hi) {
    if (g_devmem_fd < 0) return 0;
    void* m = mmap(NULL, 4096, PROT_READ, MAP_SHARED,
                   g_devmem_fd, (off_t)DPU_CTRL_BASE);
    if (m == MAP_FAILED) { perror("[ctrl_reg] mmap"); return 0; }
    volatile uint32_t* r = (volatile uint32_t*)m;
    uint64_t val = ((uint64_t)r[off_hi/4] << 32) | r[off_lo/4];
    munmap(m, 4096);
    return val;
}

static void cache_instr_address() {
    uint64_t pfn = read_ctrl_reg64(OFF_INSTR_LO, OFF_INSTR_HI);
    g_instr_phys = pfn << 12;   // PFN × 4096 = CPU physical byte address
    if (g_instr_phys)
        sim_log("[MBU][DDR4] Instructions: reg_val=0x%lX  phys=0x%016lX  size=%zu B\n",
                pfn, g_instr_phys, DDR4_INSTR_SIZE);
    else
        fprintf(stderr, "[MBU][DDR4] Warning: instruction addr=0 after baseline.\n");
}

static void cache_weights_address() {
    g_weights_phys = read_ctrl_reg64(OFF_BASE0_LO, OFF_BASE0_HI);
    if (g_weights_phys)
        sim_log("[MBU][DDR4] Weights:      phys=0x%016lX  size=%zu B\n",
                g_weights_phys, DDR4_WEIGHT_SIZE);
    else
        fprintf(stderr, "[MBU][DDR4] Warning: weights addr=0 after baseline.\n");
}

static uint64_t read_output_address() {
    return read_ctrl_reg64(OFF_BASE3_LO, OFF_BASE3_HI);
}

// ─────────────────────────────────────────────────────────────────────────────
// DDR4 BIT FLIP / RESTORE
// ─────────────────────────────────────────────────────────────────────────────
struct FlipInfo { size_t offset; int bit; uint8_t before; uint8_t after; };

// Select k unique random positions from [0, n).
// Uses Fisher-Yates partial shuffle for large k (k >= n/4) to avoid
// excessive rejection-sampling collisions. Rejection sampling for small k.
static vector<size_t> select_positions(size_t n, int k, mt19937& rng) {
    // Clamp k to region size (can't flip more unique bytes than region has)
    int actual_k = min((int)n, k);
    vector<size_t> positions;
    positions.reserve(actual_k);

    if (actual_k >= (int)(n / 4)) {
        // Fisher-Yates partial shuffle — efficient when k is a large fraction of n
        vector<size_t> indices(n);
        iota(indices.begin(), indices.end(), (size_t)0);
        for (int i = 0; i < actual_k; i++) {
            uniform_int_distribution<size_t> d(i, n - 1);
            size_t j = d(rng);
            swap(indices[i], indices[j]);
        }
        positions.assign(indices.begin(), indices.begin() + actual_k);
    } else {
        // Rejection sampling — efficient when k << n
        set<size_t> used;
        uniform_int_distribution<size_t> bdist(0, n - 1);
        int tries = 0;
        while ((int)positions.size() < actual_k && tries < actual_k * 20) {
            size_t off = bdist(rng);
            if (!used.count(off)) { used.insert(off); positions.push_back(off); }
            tries++;
        }
    }
    return positions;
}

// Flip k bits in DDR4 physical region via /dev/mem.
static vector<FlipInfo> flip_ddr4_bits(uint64_t phys_base, size_t region_size,
                                        int k, mt19937& rng, bool verbose,
                                        const char* tag) {
    vector<FlipInfo> flips;
    if (g_devmem_fd < 0 || phys_base == 0 || region_size == 0 || k <= 0)
        return flips;

    uint64_t pg_base = phys_base & ~(uint64_t)4095;
    size_t   adj     = (size_t)(phys_base - pg_base);
    size_t   map_sz  = region_size + adj;

    void* m = mmap(NULL, map_sz, PROT_READ|PROT_WRITE, MAP_SHARED,
                   g_devmem_fd, (off_t)pg_base);
    if (m == MAP_FAILED) {
        fprintf(stderr, "[MBU][flip_ddr4][%s] mmap failed phys=0x%lX: ", tag, phys_base);
        perror(""); return flips;
    }
    uint8_t* base = (uint8_t*)m + adj;

    uniform_int_distribution<int> bitd(0, 7);
    auto positions = select_positions(region_size, k, rng);

    for (size_t off : positions) {
        int bit = bitd(rng);
        uint8_t orig = base[off];
        base[off] ^= (uint8_t)(1u << bit);
        flips.push_back({off, bit, orig, base[off]});
        if (verbose)
            sim_log("  [MBU][DDR4][%s] phys=0x%016lX  off=%7zu  bit%d  0x%02X->0x%02X\n",
                    tag, phys_base + off, off, bit, orig, base[off]);
    }
    munmap(m, map_sz);
    return flips;
}

// Restore DDR4 bits via bookkeeping: write f.before back to each flipped byte.
// Mandatory for INSTRUCTIONS and WEIGHTS — both persist across runs.
// NOT needed for BUFFERS output — overwritten on each execute_async call.
static void restore_ddr4_bits(uint64_t phys_base, const vector<FlipInfo>& flips) {
    if (g_devmem_fd < 0 || flips.empty()) return;
    for (auto& f : flips) {
        uint64_t addr = phys_base + f.offset;
        uint64_t pg   = addr & ~(uint64_t)4095;
        size_t   adj  = (size_t)(addr - pg);
        void* m = mmap(NULL, adj + 1, PROT_READ|PROT_WRITE, MAP_SHARED,
                       g_devmem_fd, (off_t)pg);
        if (m == MAP_FAILED) continue;
        ((uint8_t*)m)[adj] = f.before;
        munmap(m, adj + 1);
    }
}

static bool read_ddr4_output(uint64_t phys_base, int8_t* dst, size_t n) {
    if (g_devmem_fd < 0 || phys_base == 0) return false;
    uint64_t pg  = phys_base & ~(uint64_t)4095;
    size_t   adj = (size_t)(phys_base - pg);
    size_t   sz  = n + adj;
    void* m = mmap(NULL, sz, PROT_READ, MAP_SHARED, g_devmem_fd, (off_t)pg);
    if (m == MAP_FAILED) return false;
    memcpy(dst, (uint8_t*)m + adj, n);
    munmap(m, sz);
    return true;
}

// ─────────────────────────────────────────────────────────────────────────────
// CPU BUFFER BIT FLIP  (INPUT_TENSOR target only — flips imgBuf pixels)
// ─────────────────────────────────────────────────────────────────────────────
static vector<FlipInfo> inject_sbu(uint8_t* base, size_t sz,
                                    int k, mt19937& rng, bool verbose,
                                    const char* tag) {
    vector<FlipInfo> flips;
    if (!base || sz == 0 || k <= 0) return flips;

    uniform_int_distribution<int> bitd(0, 7);
    auto positions = select_positions(sz, k, rng);

    for (size_t off : positions) {
        int bit = bitd(rng);
        uint8_t orig = base[off];
        base[off] ^= (uint8_t)(1u << bit);
        flips.push_back({off, bit, orig, base[off]});
        if (verbose)
            sim_log("  [MBU][CPU][%s] offset=%7zu  bit%d  0x%02X->0x%02X\n",
                    tag, off, bit, orig, base[off]);
    }
    return flips;
}

// ─────────────────────────────────────────────────────────────────────────────
// STRUCTS
// ─────────────────────────────────────────────────────────────────────────────
struct ImageEntry { string path; string name; int ground_truth = -1; };

struct BaselineResult {
    string image_name, image_path;
    int    ground_truth_class = -1;
    string ground_truth_name;
    int    baseline_class     = -1;
    string baseline_name;
    float  baseline_prob      = 0.f;
    bool   valid              = false;
};

struct RunResultMBU {
    string      image_name;
    int         k_bits        = 0;
    FaultTarget target_used   = FaultTarget::INPUT_TENSOR;
    string      injection_type = "MBU";           // always "MBU"
    int    ground_truth_class = -1;
    string ground_truth_name;
    int    baseline_class     = -1;
    string baseline_name;
    float  baseline_prob      = 0.f;
    int    faulty_class[3]  = {-1,-1,-1};
    float  faulty_prob[3]   = {0,0,0};
    string faulty_name[3];
    bool  correctly_classified = false;
    float prob_drop            = 0.f;
    bool  timeout              = false;
    bool  crash                = false;
    bool  output_anomaly       = false;
    uint64_t fault_addr        = 0;
    size_t   fault_byte_offset = 0;
    int      fault_bit         = 0;
};

struct SimConfig {
    string      model_path;
    string      val_folder;
    vector<int> bit_counts;
    FaultTarget target    = FaultTarget::INPUT_TENSOR;
    bool        verbose   = false;
    string      base_name = "mbu_results";
};

struct AccuracyRow {
    int   bits;
    int   total_images;
    int   baseline_correct;
    float baseline_accuracy_pct;
    int   correctly_classified;
    int   misclassified;
    float accuracy_pct;
};

// ─────────────────────────────────────────────────────────────────────────────
// FILESYSTEM HELPERS
// ─────────────────────────────────────────────────────────────────────────────
static void mkdirp(const string& path) {
    string tmp = path;
    for (size_t i = 1; i < tmp.size(); i++) {
        if (tmp[i] == '/') { tmp[i] = '\0'; mkdir(tmp.c_str(), 0755); tmp[i] = '/'; }
    }
    mkdir(tmp.c_str(), 0755);
}
static void clear_dir(const string& path) {
    DIR* d = opendir(path.c_str()); if (!d) return;
    struct dirent* e;
    while ((e = readdir(d)) != nullptr) {
        if (string(e->d_name) == "." || string(e->d_name) == "..") continue;
        string fp = path + "/" + e->d_name;
        struct stat s; lstat(fp.c_str(), &s);
        if (S_ISREG(s.st_mode)) unlink(fp.c_str());
    }
    closedir(d);
}

// Prepares:
//   target_dir = ./FaultResults/<base>/<target>/
//   csv_dir    = ./FaultResults/<base>/<target>/csv/
// Returns target_dir. csv_dir = target_dir + "/csv"
static string prepare_dirs(const string& base_name, FaultTarget target) {
    string tdir = "./FaultResults/" + base_name + "/" + targetDirName(target);
    string cdir = tdir + "/csv";
    mkdirp(cdir);
    // Clear stale files from BOTH tdir and csv/ subdir.
    // clear_dir only removes regular files — the csv/ subdir itself is preserved.
    clear_dir(tdir);   // removes any old CSVs sitting directly in target folder
    clear_dir(cdir);   // removes old per-bit CSVs from csv/ subfolder
    printf("[Dir] Target dir : %s/\n", tdir.c_str());
    printf("[Dir] CSV dir    : %s/\n", cdir.c_str());
    return tdir;
}

// ─────────────────────────────────────────────────────────────────────────────
// DATA LOADING
// ─────────────────────────────────────────────────────────────────────────────
static map<string,int> LoadSynsets(const string& path) {
    map<string,int> m; ifstream f(path);
    if (!f) { fprintf(stderr, "[Warn] synset.txt not found: %s\n", path.c_str()); return m; }
    string line; int idx = 0;
    while (getline(f, line)) { if (!line.empty()) m[line] = idx; idx++; }
    return m;
}
static void ListImagesWithGroundTruth(const string& val_dir,
                                       const map<string,int>& synset_to_idx,
                                       vector<ImageEntry>& entries) {
    entries.clear();
    struct stat s; lstat(val_dir.c_str(), &s);
    if (!S_ISDIR(s.st_mode)) {
        fprintf(stderr, "[Error] Not a directory: %s\n", val_dir.c_str()); exit(1);
    }
    DIR* top = opendir(val_dir.c_str());
    if (!top) { fprintf(stderr, "[Error] Cannot open: %s\n", val_dir.c_str()); exit(1); }
    struct dirent* cls_e;
    while ((cls_e = readdir(top)) != nullptr) {
        if (cls_e->d_name[0] == '.') continue;
        string synset   = cls_e->d_name;
        string cls_path = val_dir + "/" + synset;
        struct stat cs; lstat(cls_path.c_str(), &cs);
        if (!S_ISDIR(cs.st_mode)) continue;
        auto it = synset_to_idx.find(synset);
        if (it == synset_to_idx.end()) {
            fprintf(stderr, "[Warn] Synset %s not in synset.txt — skip\n", synset.c_str());
            continue;
        }
        int gt_class = it->second;
        DIR* sub = opendir(cls_path.c_str()); if (!sub) continue;
        struct dirent* img_e;
        while ((img_e = readdir(sub)) != nullptr) {
            if (img_e->d_type == DT_REG || img_e->d_type == DT_UNKNOWN) {
                string n = img_e->d_name; if (n.size() < 4) continue;
                string ext = n.substr(n.find_last_of('.') + 1);
                transform(ext.begin(), ext.end(), ext.begin(), ::tolower);
                if (ext == "jpg" || ext == "jpeg" || ext == "png")
                    entries.push_back({cls_path + "/" + n, synset + "/" + n, gt_class});
            }
        }
        closedir(sub);
    }
    closedir(top);
    sort(entries.begin(), entries.end(),
         [](const ImageEntry& a, const ImageEntry& b){ return a.name < b.name; });
}
static void LoadWords(const string& path, vector<string>& kinds) {
    kinds.clear(); ifstream f(path);
    if (!f) { fprintf(stderr, "[Error] Cannot open: %s\n", path.c_str()); exit(1); }
    string line; while (getline(f, line)) kinds.push_back(line);
}

// ─────────────────────────────────────────────────────────────────────────────
// PREPROCESSING / POSTPROCESSING
// ─────────────────────────────────────────────────────────────────────────────
static void preprocess_image(const Mat& src, int8_t* dst,
                              int inH, int inW, float scale) {
    static const float mean[3] = {104.f, 107.f, 123.f};
    Mat rsz; resize(src, rsz, Size(inW, inH), 0, 0, INTER_LINEAR);
    for (int h = 0; h < inH; h++)
        for (int w = 0; w < inW; w++)
            for (int c = 0; c < 3; c++) {
                float v = ((float)rsz.at<Vec3b>(h,w)[c] - mean[c]) * scale;
                dst[h*inW*3+w*3+c] = (int8_t)max(-128.f, min(127.f, v));
            }
}
static void CPUCalcSoftmax(const int8_t* d, int sz, float* out, float scale) {
    double sum = 0.0;
    for (int i = 0; i < sz; i++) { out[i] = expf((float)d[i] * scale); sum += out[i]; }
    for (int i = 0; i < sz; i++) out[i] /= (float)sum;
}
static vector<int> topk(const float* p, int sz, int k) {
    vector<int> idx(sz); iota(idx.begin(), idx.end(), 0);
    partial_sort(idx.begin(), idx.begin()+k, idx.end(),
                 [&](int a, int b){ return p[a] > p[b]; });
    idx.resize(k); return idx;
}

// ─────────────────────────────────────────────────────────────────────────────
// INFERENCE
// ─────────────────────────────────────────────────────────────────────────────
struct InferenceResult {
    bool  ok        = false;
    bool  exception = false;
    int   top1      = -1;
    float top1_prob = 0.f;
    int   top_k[TOP_K]      = {};
    float top_k_prob[TOP_K] = {};
};

static InferenceResult run_inference(vart::Runner* runner,
                                      int8_t* imgBuf, int inSz, int inH, int inW,
                                      int8_t* fcBuf,  int outSz, float out_scale,
                                      const xir::Tensor* inT,
                                      const xir::Tensor* outT) {
    InferenceResult R;
    auto idims = inT->get_shape();  idims[0] = 1;
    auto odims = outT->get_shape(); odims[0] = 1;
    vector<unique_ptr<vart::TensorBuffer>> ib, ob;
    vector<shared_ptr<xir::Tensor>> bt;
    bt.push_back(shared_ptr<xir::Tensor>(xir::Tensor::create(
        inT->get_name(), idims, xir::DataType{xir::DataType::XINT, 8u})));
    ib.push_back(make_unique<CpuFlatTensorBuffer>(imgBuf, bt.back().get()));
    bt.push_back(shared_ptr<xir::Tensor>(xir::Tensor::create(
        outT->get_name(), odims, xir::DataType{xir::DataType::XINT, 8u})));
    ob.push_back(make_unique<CpuFlatTensorBuffer>(fcBuf, bt.back().get()));
    vector<vart::TensorBuffer*> ip = {ib[0].get()}, op = {ob[0].get()};
    try {
        auto job = runner->execute_async(ip, op);
        runner->wait(job.first, -1);
    } catch (...) { R.exception = true; return R; }
    vector<float> sm(outSz);
    CPUCalcSoftmax(fcBuf, outSz, sm.data(), out_scale);
    auto tk = topk(sm.data(), outSz, TOP_K);
    R.top1 = tk[0]; R.top1_prob = sm[tk[0]];
    for (int i = 0; i < TOP_K; i++) { R.top_k[i] = tk[i]; R.top_k_prob[i] = sm[tk[i]]; }
    R.ok = true;
    return R;
}

// ─────────────────────────────────────────────────────────────────────────────
// BASELINE
// ─────────────────────────────────────────────────────────────────────────────
static BaselineResult compute_baseline(vart::Runner* runner,
                                        const ImageEntry& entry,
                                        const vector<string>& kinds) {
    BaselineResult B;
    B.image_name         = entry.name;
    B.image_path         = entry.path;
    B.ground_truth_class = entry.ground_truth;
    B.ground_truth_name  = (entry.ground_truth >= 0 && entry.ground_truth < (int)kinds.size())
                            ? kinds[entry.ground_truth] : "?";
    auto outT  = runner->get_output_tensors();
    auto inT   = runner->get_input_tensors();
    float in_sc  = get_input_scale(inT[0]);
    float out_sc = get_output_scale(outT[0]);
    int outSz = shapes.outTensorList[0].size;
    int inSz  = shapes.inTensorList[0].size;
    int inH   = shapes.inTensorList[0].height;
    int inW   = shapes.inTensorList[0].width;
    vector<int8_t> imgBuf(inSz, 0), fcBuf(outSz, 0);
    Mat raw = imread(entry.path);
    if (raw.empty()) {
        sim_log("[MBU][Baseline] Cannot read: %s\n", entry.path.c_str()); return B;
    }
    preprocess_image(raw, imgBuf.data(), inH, inW, in_sc);
    auto IR = run_inference(runner, imgBuf.data(), inSz, inH, inW,
                            fcBuf.data(), outSz, out_sc, inT[0], outT[0]);
    if (!IR.ok) {
        sim_log("[MBU][Baseline] Inference failed: %s\n", B.image_name.c_str()); return B;
    }
    B.baseline_class = IR.top1;
    B.baseline_prob  = IR.top1_prob;
    B.baseline_name  = (IR.top1 >= 0 && IR.top1 < (int)kinds.size()) ? kinds[IR.top1] : "?";
    B.valid = true;
    return B;
}

// ─────────────────────────────────────────────────────────────────────────────
// SINGLE FAULTY RUN
// ─────────────────────────────────────────────────────────────────────────────
static bool perform_faulty_run(vart::Runner* runner,
                                const vector<int8_t>& imgBuf,
                                const BaselineResult& B,
                                const vector<string>& kinds,
                                FaultTarget target, int k, bool verbose,
                                mt19937& rng, RunResultMBU& RES) {
    FaultTarget eff = target;
    if (eff == FaultTarget::ALL) {
        static const FaultTarget pool[] = {
            FaultTarget::INSTRUCTIONS, FaultTarget::WEIGHTS,
            FaultTarget::INPUT_TENSOR, FaultTarget::BUFFERS};
        eff = pool[rng() % 4];
    }

    RES.k_bits             = k;
    RES.target_used        = eff;
    RES.injection_type     = "MBU";
    RES.image_name         = B.image_name;
    RES.ground_truth_class = B.ground_truth_class;
    RES.ground_truth_name  = B.ground_truth_name;
    RES.baseline_class     = B.baseline_class;
    RES.baseline_name      = B.baseline_name;
    RES.baseline_prob      = B.baseline_prob;

    auto outT    = runner->get_output_tensors();
    auto inT     = runner->get_input_tensors();
    float out_sc = get_output_scale(outT[0]);
    int outSz    = shapes.outTensorList[0].size;
    int inSz     = shapes.inTensorList[0].size;
    int inH      = shapes.inTensorList[0].height;
    int inW      = shapes.inTensorList[0].width;

    vector<int8_t> img(imgBuf);
    vector<int8_t> fcBuf(outSz, 0);

    auto fill_result = [&](const InferenceResult& IR) {
        if (IR.exception) { RES.crash = true; return; }
        for (int i = 0; i < 3; i++) {
            RES.faulty_class[i] = IR.top_k[i];
            RES.faulty_prob[i]  = IR.top_k_prob[i];
            RES.faulty_name[i]  = (IR.top_k[i] >= 0 && IR.top_k[i] < (int)kinds.size())
                                   ? kinds[IR.top_k[i]] : "?";
        }
        RES.correctly_classified = (IR.top1 == B.ground_truth_class);
        RES.prob_drop = B.baseline_prob - IR.top1_prob;
    };

    // ── INSTRUCTIONS ─────────────────────────────────────────────────────────
    if (eff == FaultTarget::INSTRUCTIONS) {
        if (g_instr_phys == 0) { RES.crash = true; return false; }

        auto flips = flip_ddr4_bits(g_instr_phys, DDR4_INSTR_SIZE,
                                     k, rng, verbose, "INSTRUCTIONS");
        if (!flips.empty()) {
            RES.fault_byte_offset = flips[0].offset;
            RES.fault_bit         = flips[0].bit;
            RES.fault_addr        = g_instr_phys + flips[0].offset;
        }

        auto IR = run_inference(runner, img.data(), inSz, inH, inW,
                                fcBuf.data(), outSz, out_sc, inT[0], outT[0]);

        // RESTORE: DPU re-fetches instructions on each execute_async call
        restore_ddr4_bits(g_instr_phys, flips);

        fill_result(IR);
        if (verbose)
            sim_log("[MBU][INSTRUCTIONS] %s k=%d gt=%d baseline=%d(%.4f) "
                    "faulty=%d(%.4f) %s\n",
                    B.image_name.c_str(), k,
                    B.ground_truth_class, B.baseline_class, B.baseline_prob,
                    IR.top1, IR.top1_prob,
                    RES.correctly_classified ? "CORRECT" : "WRONG");
        return true;
    }

    // ── WEIGHTS ──────────────────────────────────────────────────────────────
    if (eff == FaultTarget::WEIGHTS) {
        if (g_weights_phys == 0) { RES.crash = true; return false; }

        auto flips = flip_ddr4_bits(g_weights_phys, DDR4_WEIGHT_SIZE,
                                     k, rng, verbose, "WEIGHTS");
        if (!flips.empty()) {
            RES.fault_byte_offset = flips[0].offset;
            RES.fault_bit         = flips[0].bit;
            RES.fault_addr        = g_weights_phys + flips[0].offset;
        }

        auto IR = run_inference(runner, img.data(), inSz, inH, inW,
                                fcBuf.data(), outSz, out_sc, inT[0], outT[0]);

        // RESTORE: weights persist in DDR4 across all runs in this experiment
        restore_ddr4_bits(g_weights_phys, flips);

        fill_result(IR);
        if (verbose)
            sim_log("[MBU][WEIGHTS] %s k=%d gt=%d baseline=%d(%.4f) "
                    "faulty=%d(%.4f) %s\n",
                    B.image_name.c_str(), k,
                    B.ground_truth_class, B.baseline_class, B.baseline_prob,
                    IR.top1, IR.top1_prob,
                    RES.correctly_classified ? "CORRECT" : "WRONG");
        return true;
    }

    // ── INPUT_TENSOR ─────────────────────────────────────────────────────────
    // Flip k bits in the local copy of the preprocessed image buffer.
    // VART DMA-copies imgBuf to DDR4 input region at DDR4_input_base + 2080
    // during execute_async(). Corrupted pixels propagate through all 50 layers.
    // Note: true intermediate feature maps (REG_1 WORKSPACE @ dpu_base1_addr)
    //       are written by DPU between layers and are not CPU-accessible.
    if (eff == FaultTarget::INPUT_TENSOR) {
        auto f = inject_sbu(reinterpret_cast<uint8_t*>(img.data()),
                            (size_t)inSz, k, rng, verbose, "INPUT_TENSOR");
        if (!f.empty()) {
            RES.fault_byte_offset = f[0].offset;
            RES.fault_bit         = f[0].bit;
        }

        auto IR = run_inference(runner, img.data(), inSz, inH, inW,
                                fcBuf.data(), outSz, out_sc, inT[0], outT[0]);
        fill_result(IR);
        if (verbose)
            sim_log("[MBU][INPUT_TENSOR] %s k=%d gt=%d baseline=%d(%.4f) "
                    "faulty=%d(%.4f) %s\n",
                    B.image_name.c_str(), k,
                    B.ground_truth_class, B.baseline_class, B.baseline_prob,
                    IR.top1, IR.top1_prob,
                    RES.correctly_classified ? "CORRECT" : "WRONG");
        return true;
    }

    // ── BUFFERS ──────────────────────────────────────────────────────────────
    // Run clean inference first, then flip the output DDR4 region.
    // dpu_base3_addr changes each run — read fresh after wait() completes.
    // No restore needed: output overwritten on every execute_async call.
    if (eff == FaultTarget::BUFFERS) {
        auto IR = run_inference(runner, img.data(), inSz, inH, inW,
                                fcBuf.data(), outSz, out_sc, inT[0], outT[0]);
        if (IR.exception) { RES.crash = true; return false; }

        uint64_t out_phys = read_output_address();
        if (out_phys != 0) {
            auto pf = flip_ddr4_bits(out_phys, DDR4_OUTPUT_SIZE,
                                      k, rng, verbose, "BUFFERS");
            if (!pf.empty()) {
                RES.fault_byte_offset = pf[0].offset;
                RES.fault_bit         = pf[0].bit;
                RES.fault_addr        = out_phys + pf[0].offset;
            }
            // Read corrupted bytes back into fcBuf for result decoding
            read_ddr4_output(out_phys, fcBuf.data(),
                             min((size_t)outSz, DDR4_OUTPUT_SIZE));
        } else {
            // Fallback: flip fcBuf directly (equivalent — 100% DDR4 match verified)
            sim_log("[MBU][BUFFERS] out_phys=0, flipping fcBuf directly\n");
            inject_sbu(reinterpret_cast<uint8_t*>(fcBuf.data()),
                       (size_t)outSz, k, rng, false, "BUFFERS_fallback");
        }

        vector<float> sm(outSz);
        CPUCalcSoftmax(fcBuf.data(), outSz, sm.data(), out_sc);
        auto tk = topk(sm.data(), outSz, 3);
        for (int i = 0; i < 3; i++) {
            RES.faulty_class[i] = tk[i];
            RES.faulty_prob[i]  = sm[tk[i]];
            RES.faulty_name[i]  = (tk[i] >= 0 && tk[i] < (int)kinds.size())
                                   ? kinds[tk[i]] : "?";
        }
        RES.correctly_classified = (tk[0] == B.ground_truth_class);
        RES.prob_drop = B.baseline_prob - sm[tk[0]];

        if (verbose)
            sim_log("[MBU][BUFFERS] %s k=%d gt=%d baseline=%d(%.4f) "
                    "faulty=%d(%.4f) %s\n",
                    B.image_name.c_str(), k,
                    B.ground_truth_class, B.baseline_class, B.baseline_prob,
                    tk[0], sm[tk[0]],
                    RES.correctly_classified ? "CORRECT" : "WRONG");
        return true;
    }

    return false;
}

// ─────────────────────────────────────────────────────────────────────────────
// CSV OUTPUT
// CSVs go into <target_dir>/csv/results_k{k}_bits.csv
// Each row includes injection_type="MBU" and target for self-description
// ─────────────────────────────────────────────────────────────────────────────
static void write_per_bit_csv(const vector<RunResultMBU>& results,
                               int k, const string& target_dir,
                               FaultTarget target) {
    string csv_dir = target_dir + "/csv";
    mkdirp(csv_dir);
    string path = csv_dir + "/results_k" + to_string(k) + "_bits.csv";
    ofstream f(path);
    if (!f) { fprintf(stderr, "[CSV] Cannot write %s\n", path.c_str()); return; }

    // Header — includes injection_type and target for per-row labelling
    f << "injection_type,target,k_bits,"
         "image_name,"
         "ground_truth_class,ground_truth_name,"
         "baseline_class,baseline_name,baseline_prob,"
         "faulty_top1,faulty_top1_name,faulty_top1_prob,"
         "faulty_top2,faulty_top2_name,faulty_top2_prob,"
         "faulty_top3,faulty_top3_name,faulty_top3_prob,"
         "correctly_classified,prob_drop,"
         "fault_addr,fault_byte_offset,fault_bit,"
         "crash\n";

    string tname = targetName(target);
    for (auto& R : results) {
        auto q = [](const string& s){ return "\"" + s + "\""; };
        f << q(R.injection_type) << ","
          << q(targetName(R.target_used)) << ","
          << R.k_bits << ","
          << q(R.image_name) << ","
          << R.ground_truth_class << "," << q(R.ground_truth_name) << ","
          << R.baseline_class     << "," << q(R.baseline_name)     << ","
          << fixed << setprecision(6) << R.baseline_prob << ","
          << R.faulty_class[0] << "," << q(R.faulty_name[0]) << "," << R.faulty_prob[0] << ","
          << R.faulty_class[1] << "," << q(R.faulty_name[1]) << "," << R.faulty_prob[1] << ","
          << R.faulty_class[2] << "," << q(R.faulty_name[2]) << "," << R.faulty_prob[2] << ","
          << (R.correctly_classified ? 1 : 0) << ","
          << R.prob_drop << ","
          << "0x" << hex << R.fault_addr << dec << ","
          << R.fault_byte_offset << ","
          << R.fault_bit << ","
          << (R.crash ? 1 : 0) << "\n";
    }
    printf("[CSV] Saved: %s\n", path.c_str());
    (void)tname;
}

static void write_accuracy_csv(const vector<AccuracyRow>& rows,
                                const string& target_dir,
                                FaultTarget target) {
    string path = target_dir + "/accuracy_summary.csv";
    ofstream f(path);
    if (!f) { fprintf(stderr, "[CSV] Cannot write %s\n", path.c_str()); return; }
    f << "injection_type,target,"
         "bits,total_images,"
         "baseline_correctly_classified,baseline_accuracy_pct,"
         "correctly_classified,misclassified,accuracy_pct\n";
    string tname = targetName(target);
    for (auto& r : rows) {
        f << "MBU," << tname << ","
          << r.bits << "," << r.total_images << ","
          << r.baseline_correct << ","
          << fixed << setprecision(2) << r.baseline_accuracy_pct << ","
          << r.correctly_classified << "," << r.misclassified << ","
          << r.accuracy_pct << "\n";
    }
    printf("[CSV] Saved: %s\n", path.c_str());
}

// ─────────────────────────────────────────────────────────────────────────────
// PYTHON PLOT SCRIPT  (reads CSVs from csv/ subfolder)
// ─────────────────────────────────────────────────────────────────────────────
static void write_plot_script(const string& target_dir,
                               const vector<int>& bit_counts,
                               FaultTarget target) {
    string path = target_dir + "/plot_results.py";
    FILE* f = fopen(path.c_str(), "w");
    if (!f) { fprintf(stderr, "[Plot] Cannot write %s\n", path.c_str()); return; }
    const char* tname = targetName(target).c_str();

    // NOTE: No f-strings used in generated Python — all substitution via
    // % formatting or string concatenation to avoid C fprintf {{ }} escaping issues.

    fputs("#!/usr/bin/env python3\n", f);
    fputs("# MBU Fault Injection - Plot Results\n", f);
    fprintf(f, "# Target : %s\n", tname);
    fputs("# Run    : python3 plot_results.py\n\n", f);
    fputs("import os, glob, pandas as pd, matplotlib.pyplot as plt\n", f);
    fputs("import matplotlib.ticker as mticker\n\n", f);
    fputs("OUTDIR    = os.path.dirname(os.path.abspath(__file__))\n", f);
    fputs("CSV_DIR   = os.path.join(OUTDIR, 'csv')\n", f);
    fputs("PLOTS_DIR = os.path.join(OUTDIR, 'plots')\n", f);
    fputs("os.makedirs(PLOTS_DIR, exist_ok=True)\n\n", f);
    fprintf(f, "TARGET = '%s'\n\n", tname);

    // ── Chart 1: accuracy vs bits ─────────────────────────────────────────────
    fputs("# Chart 1: Accuracy vs Bits Flipped\n", f);
    fputs("acc_path = os.path.join(OUTDIR, 'accuracy_summary.csv')\n", f);
    fputs("if os.path.exists(acc_path):\n", f);
    fputs("    df_acc   = pd.read_csv(acc_path)\n", f);
    fputs("    base_acc = df_acc['baseline_accuracy_pct'].iloc[0]\n", f);
    fputs("    x_labels = ['0 (baseline)'] + df_acc['bits'].astype(str).tolist()\n", f);
    fputs("    acc_vals = [base_acc] + df_acc['accuracy_pct'].tolist()\n", f);
    fputs("    colors   = ['forestgreen'] + ['steelblue'] * len(df_acc)\n", f);
    fputs("    fig_w    = max(10, len(x_labels) * 0.9)\n", f);
    fputs("    fig, ax  = plt.subplots(figsize=(fig_w, 6))\n", f);
    fputs("    bars = ax.bar(x_labels, acc_vals, color=colors,\n", f);
    fputs("                  edgecolor='black', width=0.65)\n", f);
    // label uses % formatting — no f-string
    fputs("    ax.axhline(base_acc, color='forestgreen', linestyle='--', linewidth=1.5,\n", f);
    fputs("               label='Baseline %.1f%%' % base_acc)\n", f);
    fputs("    ax.set_xlabel('Bits Flipped (k)', fontsize=12)\n", f);
    fputs("    ax.set_ylabel('Accuracy (%)', fontsize=12)\n", f);
    // title uses string concatenation — no f-string
    fputs("    ax.set_title('MBU Fault Injection -- ' + TARGET + '\\nAccuracy vs Bit Count',\n", f);
    fputs("                 fontsize=13, fontweight='bold')\n", f);
    fputs("    ax.set_ylim(0, 110)\n", f);
    fputs("    ax.yaxis.set_major_formatter(mticker.PercentFormatter())\n", f);
    fputs("    ax.legend(fontsize=10)\n", f);
    fputs("    plt.xticks(rotation=45, ha='right', fontsize=9)\n", f);
    // data labels use % formatting — no f-string
    fputs("    for bar, v in zip(bars, acc_vals):\n", f);
    fputs("        ax.text(bar.get_x() + bar.get_width()/2, v + 1,\n", f);
    fputs("                '%.1f%%' % v, ha='center', va='bottom', fontsize=7)\n", f);
    fputs("    plt.tight_layout()\n", f);
    fputs("    out_path = os.path.join(OUTDIR, 'plot_accuracy_vs_bits.png')\n", f);
    fputs("    plt.savefig(out_path, dpi=150)\n", f);
    fputs("    plt.close()\n", f);
    // print uses string concatenation — no f-string
    fputs("    print('[Plot] Saved: ' + out_path)\n\n", f);

    // ── Chart 2: prob drop per image, one chart per k ─────────────────────────
    // All saved to PLOTS_DIR to avoid overwrite and keep folder clean.
    fputs("# Chart 2: Prob Drop per Image (one chart per k, saved to plots/ subfolder)\n", f);
    fputs("csv_files = sorted(glob.glob(os.path.join(CSV_DIR, 'results_k*_bits.csv')))\n", f);
    fputs("for csv_path in csv_files:\n", f);
    fputs("    k_str = os.path.basename(csv_path).replace('results_k', '').replace('_bits.csv', '')\n", f);
    fputs("    try:\n", f);
    fputs("        k_val = int(k_str)\n", f);
    fputs("    except ValueError:\n", f);
    fputs("        continue\n", f);
    fputs("    df = pd.read_csv(csv_path)\n", f);
    fputs("    df = df[df['crash'] == 0].copy()\n", f);
    fputs("    if df.empty:\n", f);
    fputs("        continue\n", f);
    fputs("    avg   = df.groupby('image_name')['prob_drop'].mean().reset_index()\n", f);
    fputs("    avg   = avg.sort_values('image_name')\n", f);
    fputs("    short = [os.path.basename(n) for n in avg['image_name']]\n", f);
    fputs("    fig_w = max(10, len(short) * 0.45)\n", f);
    fputs("    fig, ax = plt.subplots(figsize=(fig_w, 5))\n", f);
    fputs("    colors  = ['tomato' if v > 0.05 else 'steelblue' for v in avg['prob_drop']]\n", f);
    fputs("    ax.bar(short, avg['prob_drop'], color=colors, edgecolor='black')\n", f);
    fputs("    ax.axhline(0, color='black', linewidth=0.8, linestyle='--')\n", f);
    fputs("    ax.set_xlabel('Image', fontsize=11)\n", f);
    fputs("    ax.set_ylabel('Avg Probability Drop', fontsize=11)\n", f);
    // title and filename use string concatenation — no f-string
    fputs("    ax.set_title('MBU -- ' + TARGET + '  k=' + str(k_val) + ' bits\\n'\n", f);
    fputs("                 'Probability Drop per Image  (red = drop > 0.05)',\n", f);
    fputs("                 fontsize=12, fontweight='bold')\n", f);
    fputs("    plt.xticks(rotation=45, ha='right', fontsize=7)\n", f);
    fputs("    plt.tight_layout()\n", f);
    fputs("    fname    = 'plot_prob_drop_k' + str(k_val) + '.png'\n", f);
    fputs("    out_path = os.path.join(PLOTS_DIR, fname)\n", f);
    fputs("    plt.savefig(out_path, dpi=150)\n", f);
    fputs("    plt.close()\n", f);
    fputs("    print('[Plot] Saved: ' + out_path)\n\n", f);

    fputs("print('[Done]')\n", f);
    fclose(f);
    printf("[Script] Plot script: %s\n", path.c_str());
}


// ─────────────────────────────────────────────────────────────────────────────
// PARSE TARGET
// ─────────────────────────────────────────────────────────────────────────────
static FaultTarget parse_target(const string& s) {
    string lo = s; transform(lo.begin(), lo.end(), lo.begin(), ::tolower);
    if (lo == "instructions")                         return FaultTarget::INSTRUCTIONS;
    if (lo == "weights")                              return FaultTarget::WEIGHTS;
    if (lo == "input_tensor" || lo == "inputtensor"
        || lo == "feature_maps" || lo == "input")    return FaultTarget::INPUT_TENSOR;
    if (lo == "buffers" || lo == "output")            return FaultTarget::BUFFERS;
    if (lo == "all")                                  return FaultTarget::ALL;
    fprintf(stderr, "[Config] Unknown target '%s', using input_tensor\n", s.c_str());
    return FaultTarget::INPUT_TENSOR;
}

// ─────────────────────────────────────────────────────────────────────────────
// MAIN
// ─────────────────────────────────────────────────────────────────────────────
int main(int argc, char* argv[]) {
    if (argc < 2) {
        printf("Usage: %s <model.xmodel> [target] [-v]\n", argv[0]);
        printf("  target: instructions | weights | input_tensor | buffers | all\n");
        return -1;
    }

    mt19937 rng(static_cast<uint32_t>(time(nullptr)) ^ (uint32_t)getpid());

    SimConfig cfg;
    cfg.model_path = argv[1];
    if (argc >= 3) cfg.target  = parse_target(argv[2]);
    cfg.verbose = (argc >= 4 && string(argv[3]) == "-v");

    // /dev/mem required for all targets except INPUT_TENSOR (which uses CPU buf)
    g_devmem_fd = open("/dev/mem", O_RDWR | O_SYNC);
    if (g_devmem_fd < 0) {
        fprintf(stderr, "[Error] Cannot open /dev/mem — must run as root.\n");
        return -1;
    }
    printf("[DDR4] /dev/mem opened (fd=%d).\n", g_devmem_fd);

    printf("\n────────────────────────────────────────────\n");
    printf("  MBU Fault Injection Simulator\n");
    printf("────────────────────────────────────────────\n\n");

    // Ask target FIRST so we can show the correct default bit counts
    if (argc < 3) {
        printf("Target [instructions/weights/input_tensor/buffers/all]\n"
               "[default: input_tensor]: ");
        fflush(stdout);
        string line; getline(cin, line);
        if (!line.empty()) cfg.target = parse_target(line);
    }

    printf("Enter train folder path [default ./train_subset]: ");
    fflush(stdout);
    { string line; getline(cin, line);
      cfg.val_folder = line.empty() ? "./train_subset" : line; }

    // Show target-appropriate default bit counts
    bool is_buffers = (cfg.target == FaultTarget::BUFFERS);
    const vector<int>& default_bits = is_buffers ? DEFAULT_BITS_BUFFERS
                                                  : DEFAULT_BITS_GENERAL;
    printf("Enter bit counts (space-separated)\n");
    printf("[default for %s: ", targetName(cfg.target).c_str());
    for (int b : default_bits) printf("%d ", b);
    printf("]: ");
    fflush(stdout);
    { string line; getline(cin, line);
      istringstream iss(line); int v;
      while (iss >> v) if (v > 0) cfg.bit_counts.push_back(v); }
    if (cfg.bit_counts.empty()) cfg.bit_counts = default_bits;
    sort(cfg.bit_counts.begin(), cfg.bit_counts.end());
    cfg.bit_counts.erase(unique(cfg.bit_counts.begin(), cfg.bit_counts.end()),
                          cfg.bit_counts.end());

    printf("Enter experiment name [default mbu_results]: ");
    fflush(stdout);
    { string line; getline(cin, line);
      if (!line.empty()) cfg.base_name = line; }
    mkdirp("./FaultResults/" + cfg.base_name);

    printf("\n[Config] model        = %s\n",  cfg.model_path.c_str());
    printf("[Config] train_folder = %s\n",    cfg.val_folder.c_str());
    printf("[Config] target       = %s\n",    targetName(cfg.target).c_str());
    printf("[Config] bits         =");
    for (int k : cfg.bit_counts) printf(" %d", k);
    printf("\n\n");

    printf("[Injection methods]\n");
    printf("  INSTRUCTIONS : DDR4 /dev/mem @ (instr_reg × 4096)   flip + restore\n");
    printf("  WEIGHTS      : DDR4 /dev/mem @ dpu_base0_addr (HP0)  flip + restore\n");
    printf("  INPUT_TENSOR : CPU imgBuf XOR  →  VART DMA  →  DDR4+2080\n");
    printf("  BUFFERS      : DDR4 /dev/mem @ dpu_base3_addr (HP0)  flip after inference\n\n");

    printf("[Output structure]\n");
    printf("  FaultResults/%s/%s/\n", cfg.base_name.c_str(),
           targetDirName(cfg.target).c_str());
    printf("    accuracy_summary.csv\n");
    printf("    plot_results.py\n");
    printf("    csv/\n");
    printf("      results_k1_bits.csv\n");
    printf("      results_k5_bits.csv  ...\n\n");

    // Open log inside target-specific folder — each target keeps its own log
    // Pre-create dir here since prepare_dirs() is called later
    string log_dir = "./FaultResults/" + cfg.base_name + "/" + targetDirName(cfg.target);
    mkdirp(log_dir);
    string logpath = log_dir + "/mbu_sim.log";
    g_logfp = fopen(logpath.c_str(), "w");
    if (!g_logfp) fprintf(stderr, "[Warn] Cannot open log %s\n", logpath.c_str());
    else printf("[Log] %s\n", logpath.c_str());

    // Load labels + synset mapping
    vector<string> kinds;
    LoadWords(wordsPath + "words.txt", kinds);
    auto synset_to_idx = LoadSynsets(wordsPath + "synset.txt");
    if (synset_to_idx.empty()) {
        fprintf(stderr, "[Error] synset.txt empty or missing\n"); return -1;
    }

    // Walk train folder
    vector<ImageEntry> entries;
    ListImagesWithGroundTruth(cfg.val_folder, synset_to_idx, entries);
    if (entries.empty()) {
        fprintf(stderr, "[Error] No images in %s\n", cfg.val_folder.c_str()); return -1;
    }
    printf("[Setup] %zu images found\n", entries.size());

    // Load model + runner
    auto graph    = xir::Graph::deserialize(cfg.model_path);
    auto subgraph = get_dpu_subgraph(graph.get());
    CHECK_EQ(subgraph.size(), 1u) << "Expected one DPU subgraph";
    auto runner_owned = vart::Runner::create_runner(subgraph[0], "run");
    vart::Runner* runner = runner_owned.get();

    auto inT  = runner->get_input_tensors();
    auto outT = runner->get_output_tensors();
    static TensorShape insh[8], outsh[8];
    shapes.inTensorList  = insh;
    shapes.outTensorList = outsh;
    getTensorShape(runner, &shapes, (int)inT.size(), (int)outT.size());

    float in_sc = get_input_scale(inT[0]);
    int inSz    = shapes.inTensorList[0].size;
    int inH     = shapes.inTensorList[0].height;
    int inW     = shapes.inTensorList[0].width;
    int outSz   = shapes.outTensorList[0].size;

    sim_log("[MBU][Setup] Input  size=%d  h=%d  w=%d\n", inSz, inH, inW);
    sim_log("[MBU][Setup] Output size=%d\n", outSz);

    // ── BASELINE PHASE ────────────────────────────────────────────────────────
    printf("[MBU][Baseline] Running clean model on %zu images...\n", entries.size());
    vector<BaselineResult> baselines;
    vector<vector<int8_t>> imgBufs;
    baselines.reserve(entries.size());
    imgBufs.reserve(entries.size());

    for (size_t i = 0; i < entries.size(); i++) {
        printf("\r[MBU][Baseline] %zu / %zu  ", i+1, entries.size()); fflush(stdout);
        baselines.push_back(compute_baseline(runner, entries[i], kinds));
        vector<int8_t> buf(inSz, 0);
        Mat raw = imread(entries[i].path);
        if (!raw.empty()) preprocess_image(raw, buf.data(), inH, inW, in_sc);
        imgBufs.push_back(move(buf));
    }
    printf("\r[MBU][Baseline] Done.                       \n");

    // Cache DDR4 addresses (populated after first execute_async in baseline)
    cache_instr_address();    // reg 0x50 × 4096 = CPU phys
    cache_weights_address();  // reg 0x60 directly (HP0, 1:1)

    int base_correct = 0, base_total = 0;
    for (auto& B : baselines) {
        if (!B.valid) continue;
        base_total++;
        if (B.baseline_class == B.ground_truth_class) base_correct++;
    }
    float base_pct = base_total > 0 ? 100.f * base_correct / base_total : 0.f;
    printf("[MBU][Baseline] Accuracy: %d/%d = %.2f%%\n",
           base_correct, base_total, base_pct);
    sim_log("[MBU][Baseline] Accuracy: %d/%d = %.2f%%\n",
            base_correct, base_total, base_pct);

    // ── FAULT INJECTION PHASE ─────────────────────────────────────────────────
    string target_dir = prepare_dirs(cfg.base_name, cfg.target);
    vector<AccuracyRow> accuracy_rows;

    for (int k : cfg.bit_counts) {
        sim_log("\n[MBU] ──── k=%d bits  target=%s ────\n",
                k, targetName(cfg.target).c_str());
        printf("\n[MBU] k=%d bits  target=%s  (%zu images)\n",
               k, targetName(cfg.target).c_str(), entries.size());

        vector<RunResultMBU> results_this_k;
        results_this_k.reserve(entries.size());
        int total_correct = 0, img_total = 0;

        for (size_t idx = 0; idx < entries.size(); idx++) {
            const BaselineResult& B = baselines[idx];
            if (!B.valid) continue;

            printf("\r  [MBU][%s] [%zu/%zu] %s  k=%d ",
                   targetName(cfg.target).c_str(),
                   idx+1, entries.size(), B.image_name.c_str(), k);
            fflush(stdout);

            RunResultMBU R;
            perform_faulty_run(runner, imgBufs[idx], B, kinds,
                               cfg.target, k, cfg.verbose, rng, R);

            for (int i = 0; i < 3; i++)
                if (R.faulty_name[i].empty() && R.faulty_class[i] >= 0
                    && R.faulty_class[i] < (int)kinds.size())
                    R.faulty_name[i] = kinds[R.faulty_class[i]];

            if (R.correctly_classified) total_correct++;
            img_total++;
            results_this_k.push_back(R);
        }
        printf("\r  [MBU][%s] Done — %d images              \n",
               targetName(cfg.target).c_str(), img_total);

        write_per_bit_csv(results_this_k, k, target_dir, cfg.target);

        float acc_pct = img_total > 0 ? 100.f * total_correct / img_total : 0.f;
        accuracy_rows.push_back({k, img_total, base_correct, base_pct,
                                 total_correct, img_total - total_correct, acc_pct});

        sim_log("[MBU][Summary] k=%-6d  target=%-14s  baseline=%.2f%%  faulty=%.2f%%\n",
                k, targetName(cfg.target).c_str(), base_pct, acc_pct);
        printf("[MBU][Summary] k=%-6d  baseline=%.2f%%  faulty=%.2f%%\n",
               k, base_pct, acc_pct);
    }

    write_accuracy_csv(accuracy_rows, target_dir, cfg.target);
    write_plot_script(target_dir, cfg.bit_counts, cfg.target);

    printf("\n────────────────────────────────────────────\n");
    printf("  [MBU] ACCURACY SUMMARY — %s\n", targetName(cfg.target).c_str());
    printf("────────────────────────────────────────────\n");
    printf("  Baseline (k=0): %d/%d = %.2f%%\n",
           base_correct, base_total, base_pct);
    for (auto& r : accuracy_rows)
        printf("  k=%-6d  %d/%d = %.2f%%  (drop = %.2f%%)\n",
               r.bits, r.correctly_classified, r.total_images,
               r.accuracy_pct, r.baseline_accuracy_pct - r.accuracy_pct);
    printf("────────────────────────────────────────────\n");
    printf("\n[MBU][Done] Results: %s/\n", target_dir.c_str());

    if (g_logfp) fclose(g_logfp);
    if (g_devmem_fd >= 0) close(g_devmem_fd);
    return 0;
}
