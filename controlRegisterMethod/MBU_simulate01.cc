/*
 * MBU_simulate.cc  –  Multi-Bit Upset (MBU) Fault Injection Simulator
 * ======================================================================
 *
 * FAULT INJECTION METHODS (per target):
 * ──────────────────────────────────────
 * INSTRUCTIONS : DIRECT DDR4 flip via /dev/mem using control register.
 *                dpu_instr_addr (reg 0x50) uses PFN encoding:
 *                CPU_phys = reg_value × 4096  (confirmed by ddr4_verify scan).
 *                Bits flipped BEFORE execute_async(), restored AFTER wait().
 *
 * WEIGHTS      : DIRECT DDR4 flip via /dev/mem at dpu_base0_addr (HP0 1:1).
 *                Address stable across runs. Flip before, restore after.
 *
 * FEATURE_MAPS : imgBuf flip before execute_async(). VART DMA copies imgBuf
 *                to DDR4 input region (REG_2) at +2080 offset (header).
 *
 * BUFFERS      : DIRECT DDR4 flip via /dev/mem at dpu_base3_addr (HP0 1:1).
 *                Address read fresh each run. Flip AFTER wait() completes.
 *
 * DDR4 ADDRESS MAP (ResNet50, ZCU104, confirmed by ddr4_verify):
 *   Reg 0x50  dpu_instr_addr  PFN encoding: reg_val × 4096 = CPU phys  742,492 B
 *   Reg 0x60  dpu_base0_addr  1:1 physical  weights REG_0 CONST        25,726,976 B
 *   Reg 0x68  dpu_base1_addr  1:1 physical  fmaps   REG_1 WORKSPACE     2,207,744 B
 *   Reg 0x70  dpu_base2_addr  1:1 physical  input   REG_2 INTERFACE       152,608 B
 *                             image data at offset +2080 within this region
 *   Reg 0x78  dpu_base3_addr  1:1 physical  output  REG_3 INTERFACE         1,008 B
 *
 * BUILD:
 *   g++ -std=c++17 -O2 -o MBU_simulate src/MBU_simulate.cc \
 *       ../common/common.cpp \
 *       -I ./src -I ../common \
 *       -I /usr/include/opencv4 -I /usr/include/vitis_ai \
 *       $(pkg-config --cflags --libs opencv4) \
 *       -lvart-runner -lxir -lglog -lpthread
 *
 * Usage:
 *   ./MBU_simulate <model.xmodel> [target] [-v]
 *   (remaining params prompted interactively)
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

// DDR4 region sizes confirmed by xir dump_bin + xir dump_reg + ddr4_verify
static const size_t DDR4_INSTR_SIZE   = 742492;     // mc_code
static const size_t DDR4_WEIGHT_SIZE  = 25726976;   // REG_0 CONST
static const size_t DDR4_OUTPUT_SIZE  = 1008;        // REG_3 INTERFACE (8B padding)

// VART prepends 2080-byte header before image pixels in DDR4 input region
static const size_t DDR4_INPUT_HDR    = 2080;

// AXI control register base + offsets (DPUCZDX8G_1, from xclbinutil)
static const uint32_t DPU_CTRL_BASE   = 0x80000000;
static const uint32_t OFF_INSTR_LO    = 0x50;   // dpu_instr_addr LO  PFN encoding (HPC0)
static const uint32_t OFF_INSTR_HI    = 0x54;   // dpu_instr_addr HI
static const uint32_t OFF_BASE0_LO    = 0x60;   // dpu_base0_addr LO  weights (HP0 1:1)
static const uint32_t OFF_BASE0_HI    = 0x64;
static const uint32_t OFF_BASE3_LO    = 0x78;   // dpu_base3_addr LO  output  (HP0 1:1)
static const uint32_t OFF_BASE3_HI    = 0x7C;

static const string wordsPath = "./";

// ─────────────────────────────────────────────────────────────────────────────
// FAULT TARGET
// ─────────────────────────────────────────────────────────────────────────────
enum class FaultTarget { INSTRUCTIONS, WEIGHTS, FEATURE_MAPS, BUFFERS, ALL };

static string targetName(FaultTarget t) {
    switch(t) {
        case FaultTarget::INSTRUCTIONS: return "INSTRUCTIONS";
        case FaultTarget::WEIGHTS:      return "WEIGHTS";
        case FaultTarget::FEATURE_MAPS: return "FEATURE_MAPS";
        case FaultTarget::BUFFERS:      return "BUFFERS";
        case FaultTarget::ALL:          return "ALL";
    }
    return "UNKNOWN";
}

static string targetDirName(FaultTarget t) {
    switch(t) {
        case FaultTarget::INSTRUCTIONS: return "instructions";
        case FaultTarget::WEIGHTS:      return "weights";
        case FaultTarget::FEATURE_MAPS: return "feature_maps";
        case FaultTarget::BUFFERS:      return "buffers";
        case FaultTarget::ALL:          return "all";
    }
    return "unknown";
}

// ─────────────────────────────────────────────────────────────────────────────
// DDR4 DIRECT ACCESS STATE
// ─────────────────────────────────────────────────────────────────────────────
// g_devmem_fd    : /dev/mem fd, opened once in main().
// g_instr_phys   : CPU physical address of instruction buffer.
//                  = dpu_instr_addr_reg × 4096  (PFN encoding, HPC0 port).
//                  Cached after first baseline inference.
// g_weights_phys : dpu_base0_addr — stable across runs (HP0, 1:1 mapping).
//                  Cached after first baseline inference.
// Output addr    : dpu_base3_addr — changes each run → read fresh per injection.
// ─────────────────────────────────────────────────────────────────────────────
static int      g_devmem_fd    = -1;
static uint64_t g_instr_phys   = 0;
static uint64_t g_weights_phys = 0;

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
// Read one 64-bit DDR4 address from two consecutive 32-bit AXI control regs.
// Maps 4KB at DPU_CTRL_BASE, reads LO+HI words, unmaps.
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

// Cache instruction DDR4 physical address.
// dpu_instr_addr uses PFN encoding: CPU_phys = reg_value × 4096.
// Confirmed by ddr4_verify scan: reg=0x1B900 → phys=0x1B900000.
// Called once after first baseline inference populates the register.
static void cache_instr_address() {
    uint64_t pfn = read_ctrl_reg64(OFF_INSTR_LO, OFF_INSTR_HI);
    g_instr_phys = pfn << 12;   // PFN × 4096
    if (g_instr_phys)
        sim_log("[DDR4] Instructions: reg_val=0x%lX  phys=0x%016lX  size=%zu B\n",
                pfn, g_instr_phys, DDR4_INSTR_SIZE);
    else
        fprintf(stderr, "[DDR4] Warning: instr address = 0 after baseline.\n");
}

// Cache weights DDR4 physical address.
// dpu_base0_addr uses 1:1 HP0 mapping — reg value IS the CPU phys address.
// Stable across runs (loaded once at create_runner()).
static void cache_weights_address() {
    g_weights_phys = read_ctrl_reg64(OFF_BASE0_LO, OFF_BASE0_HI);
    if (g_weights_phys)
        sim_log("[DDR4] Weights:      phys=0x%016lX  size=%zu B\n",
                g_weights_phys, DDR4_WEIGHT_SIZE);
    else
        fprintf(stderr, "[DDR4] Warning: weights address = 0 after baseline.\n");
}

// Read output DDR4 address fresh (changes each run).
// dpu_base3_addr, 1:1 HP0 mapping.
static uint64_t read_output_address() {
    return read_ctrl_reg64(OFF_BASE3_LO, OFF_BASE3_HI);
}

// ─────────────────────────────────────────────────────────────────────────────
// DDR4 BIT FLIP / RESTORE  (via /dev/mem, HP0 and HPC0 regions)
// ─────────────────────────────────────────────────────────────────────────────
struct FlipInfo { size_t offset; int bit; uint8_t before; uint8_t after; };

// Flip k random bits in DDR4 physical region [phys_base, phys_base+region_size).
// Returns FlipInfo list for restoration.
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
        fprintf(stderr, "[flip_ddr4][%s] mmap failed phys=0x%lX: ", tag, phys_base);
        perror(""); return flips;
    }
    uint8_t* base = (uint8_t*)m + adj;

    uniform_int_distribution<size_t> bdist(0, region_size - 1);
    uniform_int_distribution<int>    bitd(0, 7);
    set<size_t> used; int tries = 0;

    while ((int)flips.size() < k && tries < k * 20) {
        size_t off = bdist(rng);
        if (used.count(off)) { tries++; continue; }
        used.insert(off);
        int bit = bitd(rng);
        uint8_t orig = base[off];
        base[off] ^= (uint8_t)(1u << bit);
        flips.push_back({off, bit, orig, base[off]});
        if (verbose)
            sim_log("  [DDR4][%s] phys=0x%016lX  off=%7zu  bit%d  0x%02X->0x%02X\n",
                    tag, phys_base + off, off, bit, orig, base[off]);
        tries++;
    }
    munmap(m, map_sz);
    return flips;
}

// Restore DDR4 bits flipped by flip_ddr4_bits().
// MUST be called after instructions and weights injection — both regions
// persist across runs and must be clean for subsequent inferences.
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

// Read n bytes from DDR4 output region into dst.
// Used after BUFFERS flip to get corrupted logits into fcBuf.
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
// CPU BUFFER BIT FLIP  (used only for FEATURE_MAPS imgBuf)
// ─────────────────────────────────────────────────────────────────────────────
static vector<FlipInfo> inject_sbu(uint8_t* base, size_t sz,
                                    int k, mt19937& rng, bool verbose,
                                    const char* tag) {
    vector<FlipInfo> flips;
    if (!base || sz == 0 || k <= 0) return flips;
    uniform_int_distribution<size_t> bdist(0, sz - 1);
    uniform_int_distribution<int>    bitdist(0, 7);
    set<size_t> used; int tries = 0;
    while ((int)flips.size() < k && tries < k * 20) {
        size_t off = bdist(rng);
        if (used.count(off)) { tries++; continue; }
        used.insert(off);
        int bit = bitdist(rng);
        uint8_t orig = base[off];
        base[off] ^= (uint8_t)(1u << bit);
        flips.push_back({off, bit, orig, base[off]});
        if (verbose)
            sim_log("  [CPU][%s] offset=%7zu bit%d 0x%02X->0x%02X\n",
                    tag, off, bit, orig, base[off]);
        tries++;
    }
    return flips;
}

// ─────────────────────────────────────────────────────────────────────────────
// STRUCTS
// ─────────────────────────────────────────────────────────────────────────────
struct ImageEntry {
    string path;
    string name;
    int    ground_truth = -1;
};

struct BaselineResult {
    string image_name;
    string image_path;
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
    FaultTarget target_used   = FaultTarget::FEATURE_MAPS;

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

    bool  timeout        = false;
    bool  crash          = false;
    bool  output_anomaly = false;

    uint64_t fault_addr        = 0;
    size_t   fault_byte_offset = 0;
    int      fault_bit         = 0;
};

struct SimConfig {
    string          model_path;
    string          val_folder;
    vector<int>     bit_counts;
    FaultTarget     target    = FaultTarget::FEATURE_MAPS;
    bool            verbose   = false;
    string          base_name = "mbu_results";
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
        if (tmp[i] == '/') {
            tmp[i] = '\0'; mkdir(tmp.c_str(), 0755); tmp[i] = '/';
        }
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

static string prepare_target_dir(const string& base_name, FaultTarget target) {
    string tdir = "./FaultResults/" + base_name + "/" + targetDirName(target);
    mkdirp(tdir);
    clear_dir(tdir);
    printf("[Dir] Output: %s\n", tdir.c_str());
    return tdir;
}

// ─────────────────────────────────────────────────────────────────────────────
// DATA LOADING
// ─────────────────────────────────────────────────────────────────────────────
static map<string,int> LoadSynsets(const string& path) {
    map<string,int> m;
    ifstream f(path);
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
                if (ext == "jpg" || ext == "jpeg" || ext == "png") {
                    entries.push_back({cls_path + "/" + n, synset + "/" + n, gt_class});
                }
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
// INFERENCE  (direct execute_async + wait, no thread/timeout wrapper)
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
    } catch (...) {
        R.exception = true; return R;
    }
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
    int outSz  = shapes.outTensorList[0].size;
    int inSz   = shapes.inTensorList[0].size;
    int inH    = shapes.inTensorList[0].height;
    int inW    = shapes.inTensorList[0].width;

    vector<int8_t> imgBuf(inSz, 0), fcBuf(outSz, 0);
    Mat raw = imread(entry.path);
    if (raw.empty()) {
        sim_log("[Baseline] Cannot read: %s\n", entry.path.c_str()); return B;
    }
    preprocess_image(raw, imgBuf.data(), inH, inW, in_sc);
    auto IR = run_inference(runner, imgBuf.data(), inSz, inH, inW,
                            fcBuf.data(), outSz, out_sc, inT[0], outT[0]);
    if (!IR.ok) {
        sim_log("[Baseline] Inference failed: %s\n", B.image_name.c_str()); return B;
    }
    B.baseline_class = IR.top1;
    B.baseline_prob  = IR.top1_prob;
    B.baseline_name  = (IR.top1 >= 0 && IR.top1 < (int)kinds.size()) ? kinds[IR.top1] : "?";
    B.valid = true;
    return B;
}

// ─────────────────────────────────────────────────────────────────────────────
// SINGLE FAULTY RUN
//
// INSTRUCTIONS : DDR4 direct flip via /dev/mem at g_instr_phys.
//                g_instr_phys = dpu_instr_addr_reg × 4096  (PFN × 4096).
//                Flip k bits BEFORE execute_async().
//                RESTORE bits AFTER wait()  (instructions persist in DDR4).
//
// WEIGHTS      : DDR4 direct flip via /dev/mem at g_weights_phys (HP0 1:1).
//                Flip k bits BEFORE execute_async().
//                RESTORE bits AFTER wait()  (weights persist in DDR4).
//
// FEATURE_MAPS : CPU imgBuf flip before execute_async().
//                VART DMA copies corrupted imgBuf to DDR4+2080 on execute_async().
//                No DDR4 direct needed — DMA copy is physically equivalent.
//
// BUFFERS      : Clean inference first. Read dpu_base3_addr fresh (changes per run).
//                Flip k bits in DDR4 output region AFTER wait().
//                Read back corrupted bytes into fcBuf for result decoding.
//                No restore needed — output overwritten each inference.
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
            FaultTarget::FEATURE_MAPS, FaultTarget::BUFFERS};
        eff = pool[rng() % 4];
    }

    RES.k_bits             = k;
    RES.target_used        = eff;
    RES.image_name         = B.image_name;
    RES.ground_truth_class = B.ground_truth_class;
    RES.ground_truth_name  = B.ground_truth_name;
    RES.baseline_class     = B.baseline_class;
    RES.baseline_name      = B.baseline_name;
    RES.baseline_prob      = B.baseline_prob;

    auto outT     = runner->get_output_tensors();
    auto inT      = runner->get_input_tensors();
    float out_sc  = get_output_scale(outT[0]);
    int outSz     = shapes.outTensorList[0].size;
    int inSz      = shapes.inTensorList[0].size;
    int inH       = shapes.inTensorList[0].height;
    int inW       = shapes.inTensorList[0].width;

    vector<int8_t> img(imgBuf);
    vector<int8_t> fcBuf(outSz, 0);

    // ── INSTRUCTIONS ─────────────────────────────────────────────────────────
    // g_instr_phys = dpu_instr_addr_reg × 4096 (PFN encoding, HPC0 port).
    // Flip k bits in DDR4 instruction region BEFORE execute_async().
    // DPU fetches (possibly re-fetches) from DDR4 on each execute_async() call.
    // RESTORE after wait() — instructions must be clean for subsequent runs.
    if (eff == FaultTarget::INSTRUCTIONS) {
        if (g_instr_phys == 0) {
            sim_log("[INSTR] g_instr_phys=0 — instruction address not cached.\n");
            RES.crash = true; return false;
        }
        auto flips = flip_ddr4_bits(g_instr_phys, DDR4_INSTR_SIZE,
                                     k, rng, verbose, "instr_ddr4");
        if (!flips.empty()) {
            RES.fault_byte_offset = flips[0].offset;
            RES.fault_bit         = flips[0].bit;
            RES.fault_addr        = g_instr_phys + flips[0].offset;
        }

        auto IR = run_inference(runner, img.data(), inSz, inH, inW,
                                fcBuf.data(), outSz, out_sc, inT[0], outT[0]);

        // RESTORE instruction DDR4 — mandatory for clean subsequent runs
        restore_ddr4_bits(g_instr_phys, flips);

        if (IR.exception) { RES.crash = true; return false; }

        for (int i = 0; i < 3; i++) {
            RES.faulty_class[i] = IR.top_k[i];
            RES.faulty_prob[i]  = IR.top_k_prob[i];
            RES.faulty_name[i]  = (IR.top_k[i] >= 0 && IR.top_k[i] < (int)kinds.size())
                                   ? kinds[IR.top_k[i]] : "?";
        }
        RES.correctly_classified = (IR.top1 == B.ground_truth_class);
        RES.prob_drop = B.baseline_prob - IR.top1_prob;

        if (verbose)
            sim_log("[%s] k=%d INSTR gt=%d base=%d(%.3f) faulty=%d(%.3f) %s\n",
                    B.image_name.c_str(), k,
                    B.ground_truth_class, B.baseline_class, B.baseline_prob,
                    IR.top1, IR.top1_prob, RES.correctly_classified?"CORRECT":"WRONG");
        return true;
    }

    // ── WEIGHTS ──────────────────────────────────────────────────────────────
    // g_weights_phys = dpu_base0_addr (HP0, 1:1 physical, stable across runs).
    // Flip k bits BEFORE execute_async(). DPU fetches corrupted weights via DMA.
    // RESTORE after wait() — weights persist across runs.
    if (eff == FaultTarget::WEIGHTS) {
        if (g_weights_phys == 0) {
            sim_log("[WEIGHTS] g_weights_phys=0 — weights address not cached.\n");
            RES.crash = true; return false;
        }
        auto flips = flip_ddr4_bits(g_weights_phys, DDR4_WEIGHT_SIZE,
                                     k, rng, verbose, "weights_ddr4");
        if (!flips.empty()) {
            RES.fault_byte_offset = flips[0].offset;
            RES.fault_bit         = flips[0].bit;
            RES.fault_addr        = g_weights_phys + flips[0].offset;
        }

        auto IR = run_inference(runner, img.data(), inSz, inH, inW,
                                fcBuf.data(), outSz, out_sc, inT[0], outT[0]);

        // RESTORE weight DDR4 — mandatory for clean subsequent runs
        restore_ddr4_bits(g_weights_phys, flips);

        if (IR.exception) { RES.crash = true; return false; }

        for (int i = 0; i < 3; i++) {
            RES.faulty_class[i] = IR.top_k[i];
            RES.faulty_prob[i]  = IR.top_k_prob[i];
            RES.faulty_name[i]  = (IR.top_k[i] >= 0 && IR.top_k[i] < (int)kinds.size())
                                   ? kinds[IR.top_k[i]] : "?";
        }
        RES.correctly_classified = (IR.top1 == B.ground_truth_class);
        RES.prob_drop = B.baseline_prob - IR.top1_prob;

        if (verbose)
            sim_log("[%s] k=%d WEIGHTS gt=%d base=%d(%.3f) faulty=%d(%.3f) %s\n",
                    B.image_name.c_str(), k,
                    B.ground_truth_class, B.baseline_class, B.baseline_prob,
                    IR.top1, IR.top1_prob, RES.correctly_classified?"CORRECT":"WRONG");
        return true;
    }

    // ── FEATURE_MAPS ─────────────────────────────────────────────────────────
    // Flip k bits in CPU imgBuf. VART DMA copies corrupted imgBuf to DDR4+2080
    // during execute_async(). Corruption propagates through all 50 ResNet layers.
    // Note: True intermediate feature maps (REG_1 WORKSPACE) are written by DPU
    // between layers and cannot be accessed from CPU between layer boundaries.
    if (eff == FaultTarget::FEATURE_MAPS) {
        auto f = inject_sbu(reinterpret_cast<uint8_t*>(img.data()),
                            (size_t)inSz, k, rng, verbose, "fmap_imgbuf");
        if (!f.empty()) {
            RES.fault_byte_offset = f[0].offset;
            RES.fault_bit         = f[0].bit;
        }

        auto IR = run_inference(runner, img.data(), inSz, inH, inW,
                                fcBuf.data(), outSz, out_sc, inT[0], outT[0]);
        if (IR.exception) { RES.crash = true; return false; }

        for (int i = 0; i < 3; i++) {
            RES.faulty_class[i] = IR.top_k[i];
            RES.faulty_prob[i]  = IR.top_k_prob[i];
            RES.faulty_name[i]  = (IR.top_k[i] >= 0 && IR.top_k[i] < (int)kinds.size())
                                   ? kinds[IR.top_k[i]] : "?";
        }
        RES.correctly_classified = (IR.top1 == B.ground_truth_class);
        RES.prob_drop = B.baseline_prob - IR.top1_prob;

        if (verbose)
            sim_log("[%s] k=%d FMAP gt=%d base=%d(%.3f) faulty=%d(%.3f) %s\n",
                    B.image_name.c_str(), k,
                    B.ground_truth_class, B.baseline_class, B.baseline_prob,
                    IR.top1, IR.top1_prob, RES.correctly_classified?"CORRECT":"WRONG");
        return true;
    }

    // ── BUFFERS ──────────────────────────────────────────────────────────────
    // Run CLEAN inference first. DPU writes correct logits to DDR4 output.
    // Read dpu_base3_addr fresh (changes each run) AFTER wait() completes.
    // Flip k bits in DDR4 output region. Read corrupted bytes into fcBuf.
    // No restore needed — output is overwritten on each inference.
    if (eff == FaultTarget::BUFFERS) {
        // Clean inference
        auto IR = run_inference(runner, img.data(), inSz, inH, inW,
                                fcBuf.data(), outSz, out_sc, inT[0], outT[0]);
        if (IR.exception) { RES.crash = true; return false; }

        // Read fresh output DDR4 address post-inference
        uint64_t out_phys = read_output_address();
        if (out_phys != 0) {
            auto flips = flip_ddr4_bits(out_phys, DDR4_OUTPUT_SIZE,
                                         k, rng, verbose, "buffers_ddr4");
            if (!flips.empty()) {
                RES.fault_byte_offset = flips[0].offset;
                RES.fault_bit         = flips[0].bit;
                RES.fault_addr        = out_phys + flips[0].offset;
            }
            // Read corrupted output back into fcBuf
            read_ddr4_output(out_phys, fcBuf.data(),
                             min((size_t)outSz, DDR4_OUTPUT_SIZE));
        } else {
            // Fallback: flip fcBuf directly (equivalent — 100% match verified)
            sim_log("[BUFFERS] out_phys=0 — flipping fcBuf directly\n");
            inject_sbu(reinterpret_cast<uint8_t*>(fcBuf.data()),
                       (size_t)outSz, k, rng, verbose, "buffers_fcbuf");
        }

        // Decode from (possibly corrupted) fcBuf
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
            sim_log("[%s] k=%d BUFFERS gt=%d base=%d(%.3f) faulty=%d(%.3f) %s\n",
                    B.image_name.c_str(), k,
                    B.ground_truth_class, B.baseline_class, B.baseline_prob,
                    tk[0], sm[tk[0]], RES.correctly_classified?"CORRECT":"WRONG");
        return true;
    }

    return false;
}

// ─────────────────────────────────────────────────────────────────────────────
// CSV OUTPUT
// ─────────────────────────────────────────────────────────────────────────────
static void write_per_bit_csv(const vector<RunResultMBU>& results,
                               int k, const string& outDir) {
    string path = outDir + "/results_k" + to_string(k) + "_bits.csv";
    ofstream f(path);
    if (!f) { fprintf(stderr, "[CSV] Cannot write %s\n", path.c_str()); return; }

    f << "image_name,"
         "ground_truth_class,ground_truth_name,"
         "baseline_class,baseline_name,baseline_prob,"
         "faulty_top1,faulty_top1_name,faulty_top1_prob,"
         "faulty_top2,faulty_top2_name,faulty_top2_prob,"
         "faulty_top3,faulty_top3_name,faulty_top3_prob,"
         "correctly_classified,prob_drop,timeout,crash\n";

    for (auto& R : results) {
        auto q = [](const string& s){ return "\"" + s + "\""; };
        f << q(R.image_name) << ","
          << R.ground_truth_class << "," << q(R.ground_truth_name) << ","
          << R.baseline_class     << "," << q(R.baseline_name)     << ","
          << fixed << setprecision(6) << R.baseline_prob << ","
          << R.faulty_class[0] << "," << q(R.faulty_name[0]) << "," << R.faulty_prob[0] << ","
          << R.faulty_class[1] << "," << q(R.faulty_name[1]) << "," << R.faulty_prob[1] << ","
          << R.faulty_class[2] << "," << q(R.faulty_name[2]) << "," << R.faulty_prob[2] << ","
          << (R.correctly_classified ? 1 : 0) << ","
          << R.prob_drop << ","
          << (R.timeout ? 1 : 0) << ","
          << (R.crash   ? 1 : 0) << "\n";
    }
    printf("[CSV] Saved: %s\n", path.c_str());
}

static void write_accuracy_csv(const vector<AccuracyRow>& rows,
                                const string& outDir) {
    string path = outDir + "/accuracy_summary.csv";
    ofstream f(path);
    if (!f) { fprintf(stderr, "[CSV] Cannot write %s\n", path.c_str()); return; }
    f << "bits,total_images,"
         "baseline_correctly_classified,baseline_accuracy_pct,"
         "correctly_classified,misclassified,accuracy_pct\n";
    for (auto& r : rows) {
        f << r.bits << "," << r.total_images << ","
          << r.baseline_correct << ","
          << fixed << setprecision(2) << r.baseline_accuracy_pct << ","
          << r.correctly_classified << "," << r.misclassified << ","
          << r.accuracy_pct << "\n";
    }
    printf("[CSV] Saved: %s\n", path.c_str());
}

// ─────────────────────────────────────────────────────────────────────────────
// PYTHON PLOT SCRIPT
// ─────────────────────────────────────────────────────────────────────────────
static void write_plot_script(const string& outDir, const vector<int>& bit_counts) {
    string path = outDir + "/plot_results.py";
    FILE* f = fopen(path.c_str(), "w");
    if (!f) { fprintf(stderr, "[Plot] Cannot write %s\n", path.c_str()); return; }

    fprintf(f, "#!/usr/bin/env python3\n");
    fprintf(f, "import os, pandas as pd, matplotlib.pyplot as plt\n");
    fprintf(f, "import matplotlib.ticker as mticker\n\n");
    fprintf(f, "OUTDIR = os.path.dirname(os.path.abspath(__file__))\n\n");

    fprintf(f, "acc_path = os.path.join(OUTDIR, 'accuracy_summary.csv')\n");
    fprintf(f, "if os.path.exists(acc_path):\n");
    fprintf(f, "    df = pd.read_csv(acc_path)\n");
    fprintf(f, "    base_acc = df['baseline_accuracy_pct'].iloc[0]\n");
    fprintf(f, "    x = ['0 (baseline)'] + df['bits'].astype(str).tolist()\n");
    fprintf(f, "    y = [base_acc] + df['accuracy_pct'].tolist()\n");
    fprintf(f, "    fig, ax = plt.subplots(figsize=(10, 5))\n");
    fprintf(f, "    ax.bar(x, y, color=['forestgreen']+['steelblue']*len(df), edgecolor='black')\n");
    fprintf(f, "    ax.axhline(base_acc, color='forestgreen', linestyle='--',\n");
    fprintf(f, "               label=f'Baseline {base_acc:.1f}%%')\n");
    fprintf(f, "    ax.set_xlabel('Bits flipped (k)'); ax.set_ylabel('Accuracy')\n");
    fprintf(f, "    ax.set_title('MBU Fault Injection: Accuracy vs Bit Count')\n");
    fprintf(f, "    ax.set_ylim(0, 105)\n");
    fprintf(f, "    ax.yaxis.set_major_formatter(mticker.PercentFormatter())\n");
    fprintf(f, "    ax.legend()\n");
    fprintf(f, "    for i, v in enumerate(y): ax.text(i, v+1, f'{v:.1f}%%', ha='center')\n");
    fprintf(f, "    plt.tight_layout()\n");
    fprintf(f, "    plt.savefig(os.path.join(OUTDIR,'plot_accuracy_vs_bits.png'), dpi=150)\n");
    fprintf(f, "    plt.close()\n");
    fprintf(f, "    print('[Plot] plot_accuracy_vs_bits.png')\n\n");

    fprintf(f, "for k in [");
    for (size_t i = 0; i < bit_counts.size(); i++)
        fprintf(f, "%d%s", bit_counts[i], i+1 < bit_counts.size() ? "," : "");
    fprintf(f, "]:\n");
    fprintf(f, "    csv = os.path.join(OUTDIR, f'results_k{k}_bits.csv')\n");
    fprintf(f, "    if not os.path.exists(csv): continue\n");
    fprintf(f, "    df = pd.read_csv(csv)\n");
    fprintf(f, "    df = df[(df.timeout==0)&(df.crash==0)]\n");
    fprintf(f, "    avg = df.groupby('image_name')['prob_drop'].mean().reset_index()\n");
    fprintf(f, "    fig, ax = plt.subplots(figsize=(max(8,len(avg)*0.8), 5))\n");
    fprintf(f, "    colors = ['tomato' if v>0.05 else 'steelblue' for v in avg.prob_drop]\n");
    fprintf(f, "    ax.bar([n[-20:] for n in avg.image_name], avg.prob_drop,\n");
    fprintf(f, "           color=colors, edgecolor='black')\n");
    fprintf(f, "    ax.set_title(f'MBU k={k} bits: Prob Drop per Image')\n");
    fprintf(f, "    plt.xticks(rotation=45, ha='right'); plt.tight_layout()\n");
    fprintf(f, "    plt.savefig(os.path.join(OUTDIR,f'plot_prob_drop_k{k}.png'),dpi=150)\n");
    fprintf(f, "    plt.close()\n");
    fprintf(f, "    print(f'[Plot] plot_prob_drop_k{k}.png')\n");

    fclose(f);
    printf("[Script] %s\n", path.c_str());
}

// ─────────────────────────────────────────────────────────────────────────────
// PARSE TARGET
// ─────────────────────────────────────────────────────────────────────────────
static FaultTarget parse_target(const string& s) {
    string lo = s; transform(lo.begin(), lo.end(), lo.begin(), ::tolower);
    if (lo == "instructions")                                return FaultTarget::INSTRUCTIONS;
    if (lo == "weights")                                     return FaultTarget::WEIGHTS;
    if (lo == "feature_maps" || lo == "featuremaps")         return FaultTarget::FEATURE_MAPS;
    if (lo == "buffers"      || lo == "output")              return FaultTarget::BUFFERS;
    if (lo == "all")                                         return FaultTarget::ALL;
    fprintf(stderr, "[Config] Unknown target '%s', using feature_maps\n", s.c_str());
    return FaultTarget::FEATURE_MAPS;
}

// ─────────────────────────────────────────────────────────────────────────────
// MAIN
// ─────────────────────────────────────────────────────────────────────────────
int main(int argc, char* argv[]) {
    if (argc < 2) {
        printf("Usage: %s <model.xmodel> [target] [-v]\n", argv[0]);
        printf("  target: instructions|weights|feature_maps|buffers|all\n");
        return -1;
    }

    mt19937 rng(static_cast<uint32_t>(time(nullptr)) ^ (uint32_t)getpid());

    SimConfig cfg;
    cfg.model_path = argv[1];
    if (argc >= 3) cfg.target  = parse_target(argv[2]);
    cfg.verbose = (argc >= 4 && string(argv[3]) == "-v");

    // Open /dev/mem for DDR4 direct access (must run as root)
    g_devmem_fd = open("/dev/mem", O_RDWR | O_SYNC);
    if (g_devmem_fd < 0) {
        fprintf(stderr, "[DDR4] Cannot open /dev/mem — must run as root.\n");
        return -1;
    }
    printf("[DDR4] /dev/mem opened (fd=%d). DDR4 direct injection enabled.\n",
           g_devmem_fd);

    // Interactive prompts
    printf("\n--------------------------------------------\n");
    printf("   MBU Fault Injection Simulator — Setup\n");
    printf("--------------------------------------------\n\n");

    printf("Enter train folder path [default ./train_subset]: ");
    fflush(stdout);
    { string line; getline(cin, line);
      cfg.val_folder = line.empty() ? "./train_subset" : line; }

    printf("Enter bit counts (space-separated) [default 1 5 10 15 20]: ");
    fflush(stdout);
    { string line; getline(cin, line);
      istringstream iss(line); int v;
      while (iss >> v) if (v > 0) cfg.bit_counts.push_back(v); }
    if (cfg.bit_counts.empty()) cfg.bit_counts = {1, 5, 10, 15, 20};
    sort(cfg.bit_counts.begin(), cfg.bit_counts.end());
    cfg.bit_counts.erase(unique(cfg.bit_counts.begin(), cfg.bit_counts.end()),
                          cfg.bit_counts.end());

    printf("Enter experiment name [default mbu_results]: ");
    fflush(stdout);
    { string line; getline(cin, line);
      if (!line.empty()) cfg.base_name = line; }
    mkdirp("./FaultResults/" + cfg.base_name);

    if (argc < 3) {
        printf("Target [instructions/weights/feature_maps/buffers/all, default feature_maps]: ");
        fflush(stdout);
        string line; getline(cin, line);
        if (!line.empty()) cfg.target = parse_target(line);
    }

    printf("\n[Config] model       = %s\n",  cfg.model_path.c_str());
    printf("[Config] train_folder= %s\n",   cfg.val_folder.c_str());
    printf("[Config] bits        =");
    for (int k : cfg.bit_counts) printf(" %d", k);
    printf("\n[Config] target      = %s\n",  targetName(cfg.target).c_str());

    printf("\n[Methods]\n");
    printf("  INSTRUCTIONS : DDR4 /dev/mem @ (dpu_instr_addr_reg × 4096)  flip+restore\n");
    printf("  WEIGHTS      : DDR4 /dev/mem @ dpu_base0_addr               flip+restore\n");
    printf("  FEATURE_MAPS : CPU imgBuf flip -> VART DMA -> DDR4+2080\n");
    printf("  BUFFERS      : DDR4 /dev/mem @ dpu_base3_addr (fresh/run)   flip post-inference\n\n");

    // Open log
    string logpath = "./FaultResults/" + cfg.base_name + "/mbu_sim.log";
    g_logfp = fopen(logpath.c_str(), "w");
    if (!g_logfp) fprintf(stderr, "[Warn] Cannot open log %s\n", logpath.c_str());

    // Load labels + synset mapping
    vector<string> kinds;
    LoadWords(wordsPath + "words.txt", kinds);
    map<string,int> synset_to_idx = LoadSynsets(wordsPath + "synset.txt");
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

    // Load model + create runner
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

    sim_log("[Setup] Input  %s  size=%d h=%d w=%d\n",
            inT[0]->get_name().c_str(), inSz, inH, inW);
    sim_log("[Setup] Output %s  size=%d\n",
            outT[0]->get_name().c_str(), shapes.outTensorList[0].size);

    // ── BASELINE PHASE ────────────────────────────────────────────────────────
    printf("[Baseline] Running clean model on %zu images...\n", entries.size());
    vector<BaselineResult> baselines;
    vector<vector<int8_t>> imgBufs;
    baselines.reserve(entries.size());
    imgBufs.reserve(entries.size());

    for (size_t i = 0; i < entries.size(); i++) {
        printf("\r[Baseline] %zu / %zu  ", i+1, entries.size()); fflush(stdout);
        baselines.push_back(compute_baseline(runner, entries[i], kinds));
        vector<int8_t> buf(inSz, 0);
        Mat raw = imread(entries[i].path);
        if (!raw.empty()) preprocess_image(raw, buf.data(), inH, inW, in_sc);
        imgBufs.push_back(move(buf));
    }
    printf("\r[Baseline] Done.                    \n");

    // Cache DDR4 addresses after baseline populates control registers
    cache_instr_address();    // reads 0x50, multiplies by 4096
    cache_weights_address();  // reads 0x60 directly (1:1 HP0)

    int base_correct = 0, base_total = 0;
    for (auto& B : baselines) {
        if (!B.valid) continue;
        base_total++;
        if (B.baseline_class == B.ground_truth_class) base_correct++;
    }
    float base_pct = base_total > 0 ? 100.f * base_correct / base_total : 0.f;
    printf("[Baseline] Accuracy: %d/%d = %.2f%%\n",
           base_correct, base_total, base_pct);
    sim_log("[Baseline] Accuracy: %d/%d = %.2f%%\n",
            base_correct, base_total, base_pct);

    // ── FAULT INJECTION PHASE ─────────────────────────────────────────────────
    vector<AccuracyRow> accuracy_rows;
    string target_dir = prepare_target_dir(cfg.base_name, cfg.target);

    for (int k : cfg.bit_counts) {
        sim_log("\n──── k=%d bits ────\n", k);
        printf("\n[Run] k=%d bits  (%zu images)\n", k, entries.size());

        vector<RunResultMBU> results_this_k;
        results_this_k.reserve(entries.size());
        int total_correct = 0, img_total = 0;

        for (size_t img_idx = 0; img_idx < entries.size(); img_idx++) {
            const BaselineResult& B = baselines[img_idx];
            if (!B.valid) continue;

            printf("\r  [%zu/%zu] %s  k=%d  ",
                   img_idx+1, entries.size(), B.image_name.c_str(), k);
            fflush(stdout);

            RunResultMBU R;
            perform_faulty_run(runner, imgBufs[img_idx], B, kinds,
                               cfg.target, k, cfg.verbose, rng, R);

            for (int i = 0; i < 3; i++)
                if (R.faulty_name[i].empty() && R.faulty_class[i] >= 0
                    && R.faulty_class[i] < (int)kinds.size())
                    R.faulty_name[i] = kinds[R.faulty_class[i]];

            if (R.correctly_classified) total_correct++;
            img_total++;
            results_this_k.push_back(R);
        }
        printf("\r  Done %d images                          \n", img_total);

        write_per_bit_csv(results_this_k, k, target_dir);

        float acc_pct = img_total > 0 ? 100.f * total_correct / img_total : 0.f;
        accuracy_rows.push_back({k, img_total, base_correct, base_pct,
                                 total_correct, img_total - total_correct, acc_pct});

        sim_log("[Summary] k=%-3d  baseline=%.2f%%  faulty=%.2f%%\n",
                k, base_pct, acc_pct);
        printf("[Summary] k=%-3d  baseline=%.2f%%  faulty=%.2f%%\n",
               k, base_pct, acc_pct);
    }

    write_accuracy_csv(accuracy_rows, target_dir);
    write_plot_script(target_dir, cfg.bit_counts);

    printf("\n────────────────────────────────────────────\n");
    printf("  ACCURACY SUMMARY\n");
    printf("────────────────────────────────────────────\n");
    printf("  Baseline (k=0): %d/%d = %.2f%%\n",
           base_correct, base_total, base_pct);
    for (auto& r : accuracy_rows)
        printf("  k=%-3d  %d/%d = %.2f%%\n",
               r.bits, r.correctly_classified, r.total_images, r.accuracy_pct);
    printf("────────────────────────────────────────────\n");

    if (g_logfp) fclose(g_logfp);
    if (g_devmem_fd >= 0) close(g_devmem_fd);

    printf("\n[Done] Results: %s/\n", target_dir.c_str());
    return 0;
}