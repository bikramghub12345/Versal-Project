/*
 * SEFI_simulate.cc  –  Single Event Functional Interrupt (SEFI) Simulator
 * =========================================================================
 * Platform : Xilinx ZCU104  |  DPUCZDX8G  |  DDR4 + /dev/mem direct access
 * Network  : ResNet50 (.xmodel)  |  Vitis-AI VART runtime
 * Author   : Bikram Maurya
 *
 * REFERENCE:
 *   Guertin, S.M., "NEPP DDR4 Radiation Evaluation FY24 Final Report",
 *   JPL/Caltech (NASA-80NM0018D0004), 2025.
 *   Section 4, Tables 1 & 2 — SEE Types in DDR4.
 *
 * ─────────────────────────────────────────────────────────────────────────────
 * SIMULATABLE SEFI MODES (11 total):
 *
 *  Spatial SEFI (DDR4 /dev/mem):
 *   1.  SEFI-row            Corrupt entire DDR4 row (~8 KB block)
 *   2.  Transient SEFI-row  Row corruption, clears on row re-read
 *   3.  SEFI-column(band)   Stripe: same column-offset every row_stride bytes
 *   4.  Transient SEFI-col  Column stripe, clears on bank/row switch
 *   5.  SEFI-block(other)   Contiguous block corruption (configurable size)
 *   6.  Transient SEFI-blk  Block clears on row/bank switch
 *
 *  Management SEFI (runner/HW control):
 *   7.  MSEFI – DDR reset req          Inject + partial DDR reinit
 *   8.  MSEFI – ctrl reset req         Inject + controller-only reset
 *   9.  MSEFI – ctrl+DDR reset req     Inject + HW reset + recreate_runner
 *  10.  MSEFI – DDR power cycle req    Inject + close/reopen /dev/mem + recreate
 *  11.  MSEFI – full power cycle req   Inject + full reset sequence
 *
 * NOT SIMULATABLE:
 *   fixable SEFI-row/col/block — require DDR4 MRS command injection (no OS API)
 *   SEL (Single Event Latchup)  — physical latchup; would damage ZCU104
 *
 * ─────────────────────────────────────────────────────────────────────────────
 * FAULT TARGETS (when target=all, instructions is excluded):
 *   weights      — dpu_base0_addr (reg 0x60)  25,726,976 B  HP0
 *   input_tensor — dpu_base2_addr (reg 0x70)  152,608 B     HP0  (+2080 B hdr)
 *                  This is the INPUT TENSOR only. True intermediate feature maps
 *                  (REG_1 workspace) are internal to the DPU and cannot be
 *                  targeted mid-inference. No "feature_maps" target exists here.
 *   buffers      — dpu_base3_addr (reg 0x78)  1,008 B       HP0  (output buffer)
 *   instructions — dpu_instr_addr (reg 0x50)  742,492 B     HPC0 (PFN encoding)
 *
 * DPU S_AXI CONTROL BASE: 0x80000000 (ZCU104 original bitstream)
 *
 * BUILD:
 *   g++ -std=c++17 -O2 -o SEFI_simulate src/SEFI_simulate.cc \
 *       ../common/common.cpp \
 *       -I ./src -I ../common \
 *       -I /usr/include/opencv4 -I /usr/include/vitis_ai \
 *       $(pkg-config --cflags --libs opencv4) \
 *       -lvart-runner -lxir -lglog -lpthread
 *
 * USAGE (must run as root for /dev/mem):
 *   ./SEFI_simulate <model.xmodel> [target] [-v]
 *   target: weights | instructions | input_tensor | buffers | all
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
#include <cstdlib>
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

// =============================================================================
// CONSTANTS
// =============================================================================
#define TOP_K 5

// DDR4 region sizes (confirmed by ddr4_verify + xir dump_reg)
static const size_t DDR4_INSTR_SIZE   = 742492;
static const size_t DDR4_WEIGHT_SIZE  = 25726976;
static const size_t DDR4_OUTPUT_SIZE  = 1008;

// DDR4 row = 8 KB (1K columns × 8 banks × 8 bits/col / 8)
static const size_t DDR4_ROW_BYTES    = 8192;

// Default column band width for SEFI-column injection (8 B = one DDR4 burst beat)
static const size_t DDR4_COL_DEFAULT  = 8;

// DPU S_AXI control base (ZCU104 original bitstream, S_AXI at 0x80000000)
static const uint64_t DPU_CTRL_BASE   = 0x80000000ULL;

// DPU control register offsets
static const uint32_t OFF_INSTR_LO    = 0x50;
static const uint32_t OFF_INSTR_HI    = 0x54;
static const uint32_t OFF_BASE0_LO    = 0x60;
static const uint32_t OFF_BASE0_HI    = 0x64;
static const uint32_t OFF_BASE2_LO    = 0x70;  // dpu_base2_addr — input tensor (REG_2)
static const uint32_t OFF_BASE2_HI    = 0x74;
static const uint32_t OFF_BASE3_LO    = 0x78;
static const uint32_t OFF_BASE3_HI    = 0x7C;

// Input DDR4 region (REG_2 INTERFACE, confirmed by ddr4_verify):
//   total region = 152,608 B; VART prepends 2080-byte header before pixel data
static const size_t DDR4_INPUT_SIZE   = 152608;
static const size_t DDR4_INPUT_HDR    = 2080;   // byte offset to first pixel in DDR4

static const string wordsPath = "./";

// =============================================================================
// SEFI MODE ENUM  (spatial + management only)
// =============================================================================
enum class SEFIMode {
    SEFI_ROW,
    TRANSIENT_SEFI_ROW,
    SEFI_COLUMN,
    TRANSIENT_SEFI_COLUMN,
    SEFI_BLOCK,
    TRANSIENT_SEFI_BLOCK,
    MSEFI_DDR_RESET,
    MSEFI_CTRL_RESET,
    MSEFI_CTRL_DDR_RESET,
    MSEFI_DDR_POWER_CYCLE,
    MSEFI_FULL_POWER_CYCLE,
};

static const char* sefi_name(SEFIMode m) {
    switch (m) {
        case SEFIMode::SEFI_ROW:               return "SEFI-row";
        case SEFIMode::TRANSIENT_SEFI_ROW:     return "transient-SEFI-row";
        case SEFIMode::SEFI_COLUMN:            return "SEFI-column";
        case SEFIMode::TRANSIENT_SEFI_COLUMN:  return "transient-SEFI-col";
        case SEFIMode::SEFI_BLOCK:             return "SEFI-block";
        case SEFIMode::TRANSIENT_SEFI_BLOCK:   return "transient-SEFI-blk";
        case SEFIMode::MSEFI_DDR_RESET:        return "MSEFI-DDR-reset";
        case SEFIMode::MSEFI_CTRL_RESET:       return "MSEFI-ctrl-reset";
        case SEFIMode::MSEFI_CTRL_DDR_RESET:   return "MSEFI-ctrl+DDR-reset";
        case SEFIMode::MSEFI_DDR_POWER_CYCLE:  return "MSEFI-DDR-pwrcycle";
        case SEFIMode::MSEFI_FULL_POWER_CYCLE: return "MSEFI-full-pwrcycle";
    }
    return "unknown";
}

// Numbered folder name — used as the on-disk directory under sefi_results/
// e.g. "01. SEFI-row", "07. MSEFI-DDR-reset"
static string sefi_folder_name(SEFIMode m) {
    switch (m) {
        case SEFIMode::SEFI_ROW:               return "01. SEFI-row";
        case SEFIMode::TRANSIENT_SEFI_ROW:     return "02. transient-SEFI-row";
        case SEFIMode::SEFI_COLUMN:            return "03. SEFI-column";
        case SEFIMode::TRANSIENT_SEFI_COLUMN:  return "04. transient-SEFI-col";
        case SEFIMode::SEFI_BLOCK:             return "05. SEFI-block";
        case SEFIMode::TRANSIENT_SEFI_BLOCK:   return "06. transient-SEFI-blk";
        case SEFIMode::MSEFI_DDR_RESET:        return "07. MSEFI-DDR-reset";
        case SEFIMode::MSEFI_CTRL_RESET:       return "08. MSEFI-ctrl-reset";
        case SEFIMode::MSEFI_CTRL_DDR_RESET:   return "09. MSEFI-ctrl+DDR-reset";
        case SEFIMode::MSEFI_DDR_POWER_CYCLE:  return "10. MSEFI-DDR-pwrcycle";
        case SEFIMode::MSEFI_FULL_POWER_CYCLE: return "11. MSEFI-full-pwrcycle";
    }
    return "00. unknown";
}

static bool is_transient(SEFIMode m) {
    return m == SEFIMode::TRANSIENT_SEFI_ROW   ||
           m == SEFIMode::TRANSIENT_SEFI_COLUMN ||
           m == SEFIMode::TRANSIENT_SEFI_BLOCK;
}

static bool is_msefi(SEFIMode m) {
    return m == SEFIMode::MSEFI_DDR_RESET        ||
           m == SEFIMode::MSEFI_CTRL_RESET       ||
           m == SEFIMode::MSEFI_CTRL_DDR_RESET   ||
           m == SEFIMode::MSEFI_DDR_POWER_CYCLE  ||
           m == SEFIMode::MSEFI_FULL_POWER_CYCLE;
}

// =============================================================================
// FAULT TARGET
// NOTE: INPUT_TENSOR = dpu_base2_addr (input tensor only).
//       True intermediate feature maps (REG_1 workspace) are internal to DPU
//       and cannot be targeted. There is NO "feature_maps" target in this code.
// =============================================================================
enum class FaultTarget { WEIGHTS, INSTRUCTIONS, INPUT_TENSOR, BUFFERS, ALL };

static string targetName(FaultTarget t) {
    switch (t) {
        case FaultTarget::WEIGHTS:      return "weights";
        case FaultTarget::INSTRUCTIONS: return "instructions";
        case FaultTarget::INPUT_TENSOR: return "input_tensor";
        case FaultTarget::BUFFERS:      return "buffers";
        case FaultTarget::ALL:          return "all";
    }
    return "unknown";
}

// =============================================================================
// GLOBALS
// =============================================================================
static int      g_devmem_fd    = -1;
static uint64_t g_instr_phys   = 0;
static uint64_t g_weights_phys = 0;
static uint64_t g_input_phys   = 0;   // dpu_base2_addr — DDR4 input tensor (REG_2)

GraphInfo shapes;

static FILE* g_logfp = nullptr;

static void sim_log(const char* fmt, ...) {
    va_list a1, a2;
    va_start(a1, fmt); vprintf(fmt, a1); va_end(a1);
    if (g_logfp) {
        va_start(a2, fmt); vfprintf(g_logfp, fmt, a2); va_end(a2);
        fflush(g_logfp);
    }
}

// =============================================================================
// FILESYSTEM HELPERS
// =============================================================================
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

static string prepare_output_dir(SEFIMode mode, const string& tname) {
    string tdir = "./FaultResults/sefi_results/" + sefi_folder_name(mode) + "/" + tname;
    mkdirp(tdir); clear_dir(tdir);
    printf("[Dir] Output: %s\n", tdir.c_str());
    return tdir;
}

// =============================================================================
// DDR4 CONTROL REGISTER ACCESS
// =============================================================================
static uint64_t read_ctrl_reg64(uint32_t off_lo, uint32_t off_hi) {
    if (g_devmem_fd < 0) return 0;
    void* m = mmap(NULL, 4096, PROT_READ, MAP_SHARED,
                   g_devmem_fd, (off_t)(uint64_t)DPU_CTRL_BASE);
    if (m == MAP_FAILED) { perror("[ctrl_reg] mmap"); return 0; }
    volatile uint32_t* r = (volatile uint32_t*)m;
    uint64_t val = ((uint64_t)r[off_hi / 4] << 32) | r[off_lo / 4];
    munmap(m, 4096);
    return val;
}

static void cache_instr_address() {
    uint64_t pfn = read_ctrl_reg64(OFF_INSTR_LO, OFF_INSTR_HI);
    g_instr_phys = pfn << 12;
    if (g_instr_phys)
        sim_log("[DDR4] Instructions: reg_val=0x%lX  phys=0x%016lX  size=%zu B\n",
                pfn, g_instr_phys, DDR4_INSTR_SIZE);
    else
        fprintf(stderr, "[DDR4] WARNING: instr_phys=0 after baseline\n");
}

static void cache_weights_address() {
    g_weights_phys = read_ctrl_reg64(OFF_BASE0_LO, OFF_BASE0_HI);
    if (g_weights_phys)
        sim_log("[DDR4] Weights:      phys=0x%016lX  size=%zu B\n",
                g_weights_phys, DDR4_WEIGHT_SIZE);
    else
        fprintf(stderr, "[DDR4] WARNING: weights_phys=0 after baseline\n");
}

static void cache_input_address() {
    // dpu_base2_addr: 1:1 physical mapping. Pixel data at +DDR4_INPUT_HDR (2080 B).
    g_input_phys = read_ctrl_reg64(OFF_BASE2_LO, OFF_BASE2_HI);
    if (g_input_phys)
        sim_log("[DDR4] Input tensor: phys=0x%016lX  total=%zu B  pixel_start=+%zu B\n",
                g_input_phys, DDR4_INPUT_SIZE, DDR4_INPUT_HDR);
    else
        fprintf(stderr, "[DDR4] WARNING: input_phys=0 after baseline\n");
}

static uint64_t read_output_address() {
    return read_ctrl_reg64(OFF_BASE3_LO, OFF_BASE3_HI);
}

// =============================================================================
// REGION FLIP RECORD
// =============================================================================
struct RegionFlip {
    uint64_t phys_base    = 0;
    size_t   region_size  = 0;
    vector<pair<size_t, uint8_t>> restores;
    size_t bytes_affected = 0;
    size_t bits_corrupted = 0;
};

static uint8_t* region_map_rw(uint64_t phys_base, size_t sz,
                               uint64_t& pg_base, size_t& adj, size_t& map_sz) {
    pg_base = phys_base & ~(uint64_t)4095;
    adj     = (size_t)(phys_base - pg_base);
    map_sz  = sz + adj;
    void* m = mmap(NULL, map_sz, PROT_READ | PROT_WRITE, MAP_SHARED,
                   g_devmem_fd, (off_t)pg_base);
    if (m == MAP_FAILED) { perror("[mmap_rw]"); return nullptr; }
    return (uint8_t*)m + adj;
}

static void restore_region(const RegionFlip& rf) {
    if (g_devmem_fd < 0 || rf.restores.empty()) return;
    uint64_t pg; size_t adj, msz;
    uint8_t* base = region_map_rw(rf.phys_base, rf.region_size, pg, adj, msz);
    if (!base) return;
    for (auto& [off, orig] : rf.restores) base[off] = orig;
    munmap(base - adj, msz);
}

// =============================================================================
// INJECTION FUNCTION A: SEFI-row / Transient SEFI-row
// ─────────────────────────────────────────────────────
// Paper (Table 1): "A SEFI where a set of bits are corrupted (e.g. by having
//   a refresh operation start without the read system ready to obtain the data)."
// Method: Pick a random DDR4-row-aligned block (8 KB) in the target region.
//   XOR every byte with a random non-zero mask → full-row corruption signature.
//   Transient: restore all bytes after inference (clears on row re-read).
// =============================================================================
static RegionFlip inject_sefi_row(uint64_t phys_base, size_t region_sz,
                                   mt19937& rng, bool verbose, const char* tag) {
    RegionFlip rf;
    rf.phys_base   = phys_base;
    rf.region_size = region_sz;
    if (g_devmem_fd < 0 || phys_base == 0 || region_sz < DDR4_ROW_BYTES) return rf;

    uint64_t pg; size_t adj, msz;
    uint8_t* base = region_map_rw(phys_base, region_sz, pg, adj, msz);
    if (!base) return rf;

    size_t n_rows = region_sz / DDR4_ROW_BYTES;
    uniform_int_distribution<size_t>  rdist(0, n_rows - 1);
    uniform_int_distribution<uint8_t> maskd(1, 255);
    size_t row_start = rdist(rng) * DDR4_ROW_BYTES;
    size_t bits = 0;

    for (size_t i = 0; i < DDR4_ROW_BYTES; i++) {
        uint8_t xmask = maskd(rng);
        uint8_t orig  = base[row_start + i];
        base[row_start + i] ^= xmask;
        rf.restores.push_back({row_start + i, orig});
        bits += (size_t)__builtin_popcount(xmask);
    }
    rf.bytes_affected = DDR4_ROW_BYTES;
    rf.bits_corrupted = bits;

    sim_log("  [%s] SEFI-row  phys=0x%016lX  row_off=%zu  size=%zu B  bits~%zu\n",
            tag, phys_base + row_start, row_start, DDR4_ROW_BYTES, bits);
    (void)verbose;
    munmap(base - adj, msz);
    return rf;
}

// =============================================================================
// INJECTION FUNCTION B: SEFI-column(band) / Transient SEFI-column
// ─────────────────────────────────────────────────────────────────
// Paper (Table 1): "A SEFI where a small number of columns manifests errors
//   in 1000s of rows." / "disrupting data during sequential operations."
// Method: Pick a random column offset in [0, row_stride - col_width].
//   For every row in the region, XOR col_width bytes at that offset.
//   Transient: restore all bytes after inference.
// =============================================================================
static RegionFlip inject_sefi_column(uint64_t phys_base, size_t region_sz,
                                      size_t row_stride, size_t col_width,
                                      mt19937& rng, bool verbose, const char* tag) {
    RegionFlip rf;
    rf.phys_base   = phys_base;
    rf.region_size = region_sz;
    if (g_devmem_fd < 0 || phys_base == 0 || region_sz < row_stride) return rf;
    if (col_width == 0 || col_width > row_stride) col_width = DDR4_COL_DEFAULT;

    uint64_t pg; size_t adj, msz;
    uint8_t* base = region_map_rw(phys_base, region_sz, pg, adj, msz);
    if (!base) return rf;

    uniform_int_distribution<size_t>  cdist(0, row_stride - col_width);
    uniform_int_distribution<uint8_t> maskd(1, 255);
    size_t col_start = cdist(rng);
    size_t n_rows    = region_sz / row_stride;
    size_t bits      = 0;

    for (size_t row = 0; row < n_rows; row++) {
        size_t base_off = row * row_stride + col_start;
        if (base_off + col_width > region_sz) break;
        for (size_t c = 0; c < col_width; c++) {
            uint8_t xmask = maskd(rng);
            uint8_t orig  = base[base_off + c];
            base[base_off + c] ^= xmask;
            rf.restores.push_back({base_off + c, orig});
            bits += (size_t)__builtin_popcount(xmask);
        }
    }
    rf.bytes_affected = n_rows * col_width;
    rf.bits_corrupted = bits;

    sim_log("  [%s] SEFI-col  col_start=%zu  col_w=%zu  stride=%zu  rows=%zu  bits~%zu\n",
            tag, col_start, col_width, row_stride, n_rows, bits);
    (void)verbose;
    munmap(base - adj, msz);
    return rf;
}

// =============================================================================
// INJECTION FUNCTION C: SEFI-block(other) / Transient SEFI-block
// ──────────────────────────────────────────────────────────────────
// Paper (Table 1): "A SEFI where a small to moderate number of addresses
//   manifests errors." / "disrupting data during sequential operations."
// Method: Pick a random start; XOR block_sz contiguous bytes with random masks.
//   Transient: restore all bytes after inference.
// =============================================================================
static RegionFlip inject_sefi_block(uint64_t phys_base, size_t region_sz,
                                     size_t block_sz, mt19937& rng,
                                     bool verbose, const char* tag) {
    RegionFlip rf;
    rf.phys_base   = phys_base;
    rf.region_size = region_sz;
    if (g_devmem_fd < 0 || phys_base == 0 || region_sz == 0) return rf;
    if (block_sz == 0) block_sz = DDR4_ROW_BYTES;
    block_sz = min(block_sz, region_sz);

    uint64_t pg; size_t adj, msz;
    uint8_t* base = region_map_rw(phys_base, region_sz, pg, adj, msz);
    if (!base) return rf;

    uniform_int_distribution<size_t>  sdist(0, region_sz - block_sz);
    uniform_int_distribution<uint8_t> maskd(1, 255);
    size_t start = sdist(rng);
    size_t bits  = 0;

    for (size_t i = 0; i < block_sz; i++) {
        uint8_t xmask = maskd(rng);
        uint8_t orig  = base[start + i];
        base[start + i] ^= xmask;
        rf.restores.push_back({start + i, orig});
        bits += (size_t)__builtin_popcount(xmask);
    }
    rf.bytes_affected = block_sz;
    rf.bits_corrupted = bits;

    sim_log("  [%s] SEFI-block  phys=0x%016lX  start=%zu  size=%zu B  bits~%zu\n",
            tag, phys_base + start, start, block_sz, bits);
    (void)verbose;
    munmap(base - adj, msz);
    return rf;
}

// =============================================================================
// PREPROCESSING / SOFTMAX / TOP-K
// =============================================================================
static void preprocess_image(const Mat& src, int8_t* dst,
                              int inH, int inW, float scale) {
    static const float mean[3] = {104.f, 107.f, 123.f};
    Mat rsz; resize(src, rsz, Size(inW, inH), 0, 0, INTER_LINEAR);
    for (int h = 0; h < inH; h++)
        for (int w = 0; w < inW; w++)
            for (int c = 0; c < 3; c++) {
                float v = ((float)rsz.at<Vec3b>(h, w)[c] - mean[c]) * scale;
                dst[h * inW * 3 + w * 3 + c] = (int8_t)max(-128.f, min(127.f, v));
            }
}

static void CPUCalcSoftmax(const int8_t* d, int sz, float* out, float scale) {
    double sum = 0.0;
    for (int i = 0; i < sz; i++) { out[i] = expf((float)d[i] * scale); sum += out[i]; }
    for (int i = 0; i < sz; i++) out[i] /= (float)sum;
}

static vector<int> topk(const float* p, int sz, int k) {
    vector<int> idx(sz); iota(idx.begin(), idx.end(), 0);
    partial_sort(idx.begin(), idx.begin() + k, idx.end(),
                 [&](int a, int b) { return p[a] > p[b]; });
    idx.resize(k); return idx;
}

// =============================================================================
// INFERENCE HELPER
// =============================================================================
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

// =============================================================================
// RUNNER RECOVERY HELPERS
// =============================================================================
static int hardware_reset_dpu() {
    sim_log("[Reset][HW] Attempting hardware DPU reset...\n");
    int fd = open("/dev/mem", O_RDWR | O_SYNC);
    if (fd >= 0) {
        void* b = mmap(NULL, 4096, PROT_READ | PROT_WRITE, MAP_SHARED,
                       fd, (off_t)(uint64_t)DPU_CTRL_BASE);
        if (b != MAP_FAILED) {
            volatile uint32_t* c = (volatile uint32_t*)b;
            c[0] = 0x00; usleep(10000);
            c[1] = 0xFF; usleep(10000);
            c[0] = 0x01; usleep(100000);
            c[0] = 0x04; usleep(10000);
            munmap(b, 4096); close(fd);
            sim_log("[Reset][HW] Done\n"); return 0;
        }
        close(fd);
    }
    FILE* fp = fopen("/sys/class/dpu/dpu0/reset", "w");
    if (fp) { fprintf(fp, "1\n"); fclose(fp); sleep(1); return 0; }
    return -1;
}

static unique_ptr<vart::Runner> recreate_runner(const xir::Subgraph* sg) {
    sim_log("[Reset][SW] Recreating runner...\n");
    try {
        auto r = vart::Runner::create_runner(sg, "run");
        sim_log("[Reset][SW] OK\n"); return r;
    } catch (const exception& e) { sim_log("[Reset][SW] FAIL: %s\n", e.what()); }
    catch (...) { sim_log("[Reset][SW] FAIL\n"); }
    return nullptr;
}

// =============================================================================
// DATA STRUCTURES
// =============================================================================
struct ImageEntry {
    string path; string name; int ground_truth = -1;
};

struct BaselineResult {
    string image_name, image_path;
    int    ground_truth_class = -1;  string ground_truth_name;
    int    baseline_class     = -1;  string baseline_name;
    float  baseline_prob      = 0.f;
    bool   valid              = false;
};

struct RunResultSEFI {
    string      image_name;
    SEFIMode    mode;
    string      mode_name;
    FaultTarget target;
    bool        transient_mode = false;

    int    ground_truth_class = -1;  string ground_truth_name;
    int    baseline_class     = -1;  string baseline_name;
    float  baseline_prob      = 0.f;

    int    faulty_class[3] = {-1, -1, -1};
    float  faulty_prob[3]  = {0, 0, 0};
    string faulty_name[3];
    bool   correctly_classified = false;
    float  prob_drop            = 0.f;

    size_t   bytes_corrupted = 0;
    size_t   bits_corrupted  = 0;
    uint64_t fault_phys_addr = 0;

    // MSEFI recovery fields
    bool   msefi_ran        = false;
    bool   msefi_recovered  = false;
    int    recovery_class   = -1;
    float  recovery_prob    = 0.f;
    string recovery_action;

    bool crash = false;
};

struct SimConfig {
    string      model_path;
    string      val_folder;
    SEFIMode    mode       = SEFIMode::SEFI_BLOCK;
    FaultTarget target     = FaultTarget::WEIGHTS;
    bool        verbose    = false;
    size_t      col_width  = DDR4_COL_DEFAULT;
    size_t      block_size = 4096;
};

struct AccuracyRow {
    string mode_name;
    int    total_images;
    int    baseline_correct;   float baseline_pct;
    int    faulty_correct;     int   faulty_wrong;  float faulty_pct;
    int    recovered_correct;  float recovery_pct;
};

// =============================================================================
// DATA LOADING
// =============================================================================
static map<string, int> LoadSynsets(const string& path) {
    map<string, int> m;
    ifstream f(path);
    if (!f) { fprintf(stderr, "[Warn] synset.txt not found: %s\n", path.c_str()); return m; }
    string line; int idx = 0;
    while (getline(f, line)) { if (!line.empty()) m[line] = idx; idx++; }
    return m;
}

static void ListImagesWithGroundTruth(const string& val_dir,
                                       const map<string, int>& synset_to_idx,
                                       vector<ImageEntry>& entries) {
    entries.clear();
    struct stat s; lstat(val_dir.c_str(), &s);
    if (!S_ISDIR(s.st_mode)) {
        fprintf(stderr, "[Error] Not a directory: %s\n", val_dir.c_str()); exit(1);
    }
    DIR* top = opendir(val_dir.c_str());
    if (!top) { fprintf(stderr, "[Error] Cannot open: %s\n", val_dir.c_str()); exit(1); }
    struct dirent* ce;
    while ((ce = readdir(top)) != nullptr) {
        if (ce->d_name[0] == '.') continue;
        string synset   = ce->d_name;
        string cls_path = val_dir + "/" + synset;
        struct stat cs; lstat(cls_path.c_str(), &cs);
        if (!S_ISDIR(cs.st_mode)) continue;
        auto it = synset_to_idx.find(synset);
        if (it == synset_to_idx.end()) {
            fprintf(stderr, "[Warn] Synset %s not in synset.txt — skip\n", synset.c_str());
            continue;
        }
        int gt = it->second;
        DIR* sub = opendir(cls_path.c_str()); if (!sub) continue;
        struct dirent* ie;
        while ((ie = readdir(sub)) != nullptr) {
            if (ie->d_type == DT_REG || ie->d_type == DT_UNKNOWN) {
                string n = ie->d_name; if (n.size() < 4) continue;
                string ext = n.substr(n.find_last_of('.') + 1);
                transform(ext.begin(), ext.end(), ext.begin(), ::tolower);
                if (ext == "jpg" || ext == "jpeg" || ext == "png")
                    entries.push_back({cls_path + "/" + n, synset + "/" + n, gt});
            }
        }
        closedir(sub);
    }
    closedir(top);
    sort(entries.begin(), entries.end(),
         [](const ImageEntry& a, const ImageEntry& b) { return a.name < b.name; });
}

static void LoadWords(const string& path, vector<string>& kinds) {
    kinds.clear(); ifstream f(path);
    if (!f) { fprintf(stderr, "[Error] Cannot open: %s\n", path.c_str()); exit(1); }
    string line; while (getline(f, line)) kinds.push_back(line);
}

// =============================================================================
// BASELINE
// =============================================================================
static BaselineResult compute_baseline(vart::Runner* runner,
                                        const ImageEntry& entry,
                                        const vector<string>& kinds) {
    BaselineResult B;
    B.image_name         = entry.name;
    B.image_path         = entry.path;
    B.ground_truth_class = entry.ground_truth;
    B.ground_truth_name  = (entry.ground_truth >= 0 && entry.ground_truth < (int)kinds.size())
                            ? kinds[entry.ground_truth] : "?";

    auto outT    = runner->get_output_tensors();
    auto inT     = runner->get_input_tensors();
    float in_sc  = get_input_scale(inT[0]);
    float out_sc = get_output_scale(outT[0]);
    int outSz    = shapes.outTensorList[0].size;
    int inSz     = shapes.inTensorList[0].size;
    int inH      = shapes.inTensorList[0].height;
    int inW      = shapes.inTensorList[0].width;

    vector<int8_t> imgBuf(inSz, 0), fcBuf(outSz, 0);
    Mat raw = imread(entry.path);
    if (raw.empty()) { sim_log("[Baseline] Cannot read: %s\n", entry.path.c_str()); return B; }
    preprocess_image(raw, imgBuf.data(), inH, inW, in_sc);

    auto IR = run_inference(runner, imgBuf.data(), inSz, inH, inW,
                            fcBuf.data(), outSz, out_sc, inT[0], outT[0]);
    if (!IR.ok) { sim_log("[Baseline] Inference failed: %s\n", B.image_name.c_str()); return B; }

    B.baseline_class = IR.top1;
    B.baseline_prob  = IR.top1_prob;
    B.baseline_name  = (IR.top1 >= 0 && IR.top1 < (int)kinds.size()) ? kinds[IR.top1] : "?";
    B.valid = true;
    return B;
}

// =============================================================================
// MSEFI RECOVERY SIMULATION
// ─────────────────────────────────────────────────────────────────────────────
// Procedure (per Guertin Table 2):
//   1. Inject SEFI-block into weights region → simulate DDR4 bad config.
//   2. Run inference → record faulty classification.
//   3. Restore all injected bytes → DDR4 stabilised after reset.
//   4. Apply mode-specific recovery action.
//   5. Run clean inference → record post-recovery classification.
// =============================================================================
static void simulate_msefi(vart::Runner*& runner,
                             const xir::Subgraph* sg,
                             vector<int8_t>& img,
                             const BaselineResult& B,
                             const vector<string>& kinds,
                             SEFIMode msefi_mode,
                             const SimConfig& cfg,
                             mt19937& rng,
                             RunResultSEFI& RES) {
    auto outT    = runner->get_output_tensors();
    auto inT     = runner->get_input_tensors();
    float out_sc = get_output_scale(outT[0]);
    int outSz    = shapes.outTensorList[0].size;
    int inSz     = shapes.inTensorList[0].size;
    int inH      = shapes.inTensorList[0].height;
    int inW      = shapes.inTensorList[0].width;

    if (g_weights_phys == 0) { RES.crash = true; return; }

    size_t blk = max(cfg.block_size, DDR4_WEIGHT_SIZE / 10);
    blk = min(blk, DDR4_WEIGHT_SIZE);
    auto rf = inject_sefi_block(g_weights_phys, DDR4_WEIGHT_SIZE,
                                blk, rng, cfg.verbose, sefi_name(msefi_mode));
    RES.bytes_corrupted = rf.bytes_affected;
    RES.bits_corrupted  = rf.bits_corrupted;
    RES.fault_phys_addr = g_weights_phys;

    vector<int8_t> fcBuf(outSz, 0);
    auto IR = run_inference(runner, img.data(), inSz, inH, inW,
                            fcBuf.data(), outSz, out_sc, inT[0], outT[0]);
    if (IR.exception) { restore_region(rf); RES.crash = true; return; }

    for (int i = 0; i < 3; i++) {
        RES.faulty_class[i] = IR.top_k[i];
        RES.faulty_prob[i]  = IR.top_k_prob[i];
        RES.faulty_name[i]  = (IR.top_k[i] >= 0 && IR.top_k[i] < (int)kinds.size())
                               ? kinds[IR.top_k[i]] : "?";
    }
    RES.correctly_classified = (IR.top1 == B.ground_truth_class);
    RES.prob_drop            = B.baseline_prob - IR.top1_prob;

    restore_region(rf);

    string action;
    switch (msefi_mode) {
        case SEFIMode::MSEFI_DDR_RESET: {
            auto nr = recreate_runner(sg);
            if (nr) runner = nr.release();
            action = "DDR_reinit(recreate_runner)";
            break;
        }
        case SEFIMode::MSEFI_CTRL_RESET: {
            auto nr = recreate_runner(sg);
            if (nr) runner = nr.release();
            action = "ctrl_reset(recreate_runner_only)";
            break;
        }
        case SEFIMode::MSEFI_CTRL_DDR_RESET: {
            hardware_reset_dpu();
            usleep(200000);
            auto nr = recreate_runner(sg);
            if (nr) runner = nr.release();
            action = "hw_reset+recreate_runner";
            break;
        }
        case SEFIMode::MSEFI_DDR_POWER_CYCLE: {
            if (g_devmem_fd >= 0) { close(g_devmem_fd); g_devmem_fd = -1; }
            usleep(100000);
            g_devmem_fd = open("/dev/mem", O_RDWR | O_SYNC);
            cache_instr_address();
            cache_weights_address();
            cache_input_address();
            auto nr = recreate_runner(sg);
            if (nr) runner = nr.release();
            action = "devmem_reopen+recreate_runner";
            break;
        }
        case SEFIMode::MSEFI_FULL_POWER_CYCLE: {
            hardware_reset_dpu();
            if (g_devmem_fd >= 0) { close(g_devmem_fd); g_devmem_fd = -1; }
            usleep(200000);
            g_devmem_fd = open("/dev/mem", O_RDWR | O_SYNC);
            cache_instr_address();
            cache_weights_address();
            cache_input_address();
            auto nr = recreate_runner(sg);
            if (nr) runner = nr.release();
            action = "full_power_cycle(hw+devmem+runner)";
            break;
        }
        default: action = "none"; break;
    }
    RES.recovery_action = action;

    auto outT2    = runner->get_output_tensors();
    auto inT2     = runner->get_input_tensors();
    float out_sc2 = get_output_scale(outT2[0]);
    int outSz2    = shapes.outTensorList[0].size;
    vector<int8_t> fcBuf2(outSz2, 0);
    auto IR2 = run_inference(runner, img.data(), inSz, inH, inW,
                             fcBuf2.data(), outSz2, out_sc2, inT2[0], outT2[0]);
    if (IR2.ok) {
        RES.recovery_class  = IR2.top1;
        RES.recovery_prob   = IR2.top1_prob;
        RES.msefi_recovered = (IR2.top1 == B.ground_truth_class);
    }
    RES.msefi_ran = true;

    if (cfg.verbose)
        sim_log("[%s] MSEFI %s  faulty=%d(%.3f)  recov=%d(%.3f)  action=%s  %s\n",
                B.image_name.c_str(), sefi_name(msefi_mode),
                IR.top1, IR.top1_prob, RES.recovery_class, RES.recovery_prob,
                action.c_str(), RES.msefi_recovered ? "RECOVERED" : "NOT_RECOVERED");
}

// =============================================================================
// SINGLE SEFI FAULTY RUN DISPATCHER
// =============================================================================
static bool perform_sefi_run(vart::Runner*& runner,
                              const xir::Subgraph* sg,
                              vector<int8_t>& imgBuf,
                              const BaselineResult& B,
                              const vector<string>& kinds,
                              const SimConfig& cfg,
                              mt19937& rng,
                              RunResultSEFI& RES) {
    SEFIMode    mode       = cfg.mode;
    FaultTarget eff_target = cfg.target;
    // NOTE: ALL is resolved by the outer loop in main; concrete target always passed here.

    RES.mode           = mode;
    RES.mode_name      = sefi_name(mode);
    RES.target         = eff_target;
    RES.image_name     = B.image_name;
    RES.transient_mode = is_transient(mode);
    RES.ground_truth_class = B.ground_truth_class;
    RES.ground_truth_name  = B.ground_truth_name;
    RES.baseline_class     = B.baseline_class;
    RES.baseline_name      = B.baseline_name;
    RES.baseline_prob      = B.baseline_prob;

    if (is_msefi(mode)) {
        simulate_msefi(runner, sg, imgBuf, B, kinds, mode, cfg, rng, RES);
        return !RES.crash;
    }

    // Get tensor info early — needed for INPUT_TENSOR size
    auto outT    = runner->get_output_tensors();
    auto inT     = runner->get_input_tensors();
    float out_sc = get_output_scale(outT[0]);
    int outSz    = shapes.outTensorList[0].size;
    int inSz     = shapes.inTensorList[0].size;
    int inH      = shapes.inTensorList[0].height;
    int inW      = shapes.inTensorList[0].width;

    // Resolve DDR4 region
    uint64_t phys   = 0;
    size_t   rgsz   = 0;
    bool use_output = (eff_target == FaultTarget::BUFFERS);

    if (eff_target == FaultTarget::WEIGHTS) {
        phys = g_weights_phys;  rgsz = DDR4_WEIGHT_SIZE;
    } else if (eff_target == FaultTarget::INSTRUCTIONS) {
        phys = g_instr_phys;    rgsz = DDR4_INSTR_SIZE;
    } else if (eff_target == FaultTarget::INPUT_TENSOR) {
        // Direct DDR4 injection into dpu_base2_addr (input tensor region).
        // Pixel data is at g_input_phys + DDR4_INPUT_HDR (2080 B VART header).
        // We write imgBuf into DDR4 first, then inject the SEFI pattern on top,
        // so the DPU reads corrupted pixel data directly from DDR4.
        phys = g_input_phys + DDR4_INPUT_HDR;
        rgsz = (size_t)inSz;   // 224×224×3 = 150528 B
    }

    if (!use_output && phys == 0) { RES.crash = true; return false; }

    vector<int8_t> img(imgBuf);
    RegionFlip     rf;

    // ── BUFFERS: run clean inference first, then corrupt output buffer ─────────
    if (use_output) {
        vector<int8_t> fcBuf(outSz, 0);
        auto IR0 = run_inference(runner, img.data(), inSz, inH, inW,
                                 fcBuf.data(), outSz, out_sc, inT[0], outT[0]);
        if (!IR0.ok) { RES.crash = true; return false; }

        uint64_t out_phys = read_output_address();
        RegionFlip rfo;
        if (out_phys != 0) {
            size_t blk = min(cfg.block_size, DDR4_OUTPUT_SIZE);
            rfo = inject_sefi_block(out_phys, DDR4_OUTPUT_SIZE,
                                    blk, rng, cfg.verbose, "buffers_post");
            uint64_t pg  = out_phys & ~(uint64_t)4095;
            size_t   adj = (size_t)(out_phys - pg);
            size_t   msz = DDR4_OUTPUT_SIZE + adj;
            void* dm = mmap(NULL, msz, PROT_READ, MAP_SHARED, g_devmem_fd, (off_t)pg);
            if (dm != MAP_FAILED) {
                memcpy(fcBuf.data(),
                       reinterpret_cast<int8_t*>((uint8_t*)dm + adj),
                       min((size_t)outSz, DDR4_OUTPUT_SIZE));
                munmap(dm, msz);
            }
            if (RES.transient_mode) restore_region(rfo);
            RES.bytes_corrupted = rfo.bytes_affected;
            RES.bits_corrupted  = rfo.bits_corrupted;
            RES.fault_phys_addr = out_phys;
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
        RES.prob_drop            = B.baseline_prob - sm[tk[0]];
        return true;
    }

    // ── INPUT_TENSOR: write clean pixels into DDR4 first, then inject ─────────
    // After this memcpy, VART's DMA from imgBuf will overwrite DDR4 again, BUT
    // we inject the SEFI pattern immediately after the memcpy and before
    // execute_async, so the injection window is guaranteed.
    if (eff_target == FaultTarget::INPUT_TENSOR && g_input_phys != 0) {
        uint64_t pg; size_t adj, msz;
        uint8_t* ddr4_in = region_map_rw(g_input_phys + DDR4_INPUT_HDR,
                                          (size_t)inSz, pg, adj, msz);
        if (ddr4_in) {
            memcpy(ddr4_in, img.data(), (size_t)inSz);
            munmap(ddr4_in - adj, msz);
        }
    }

    // ── Spatial injection into DDR4 ───────────────────────────────────────────
    switch (mode) {
        case SEFIMode::SEFI_ROW:
        case SEFIMode::TRANSIENT_SEFI_ROW:
            rf = inject_sefi_row(phys, rgsz, rng, cfg.verbose, sefi_name(mode));
            break;
        case SEFIMode::SEFI_COLUMN:
        case SEFIMode::TRANSIENT_SEFI_COLUMN:
            rf = inject_sefi_column(phys, rgsz, DDR4_ROW_BYTES, cfg.col_width,
                                    rng, cfg.verbose, sefi_name(mode));
            break;
        case SEFIMode::SEFI_BLOCK:
        case SEFIMode::TRANSIENT_SEFI_BLOCK:
            rf = inject_sefi_block(phys, rgsz, cfg.block_size,
                                   rng, cfg.verbose, sefi_name(mode));
            break;
        default: break;
    }

    RES.bytes_corrupted = rf.bytes_affected;
    RES.bits_corrupted  = rf.bits_corrupted;
    RES.fault_phys_addr = phys + (rf.restores.empty() ? 0 : rf.restores[0].first);

    // ── Run faulty inference ──────────────────────────────────────────────────
    vector<int8_t> fcBuf(outSz, 0);
    auto IR = run_inference(runner, img.data(), inSz, inH, inW,
                            fcBuf.data(), outSz, out_sc, inT[0], outT[0]);

    // ── Transient: restore DDR4 bytes before capturing result ─────────────────
    if (RES.transient_mode) restore_region(rf);

    if (IR.exception) { RES.crash = true; return false; }

    for (int i = 0; i < 3; i++) {
        RES.faulty_class[i] = IR.top_k[i];
        RES.faulty_prob[i]  = IR.top_k_prob[i];
        RES.faulty_name[i]  = (IR.top_k[i] >= 0 && IR.top_k[i] < (int)kinds.size())
                               ? kinds[IR.top_k[i]] : "?";
    }
    RES.correctly_classified = (IR.top1 == B.ground_truth_class);
    RES.prob_drop            = B.baseline_prob - IR.top1_prob;

    if (cfg.verbose)
        sim_log("[%s] %s  gt=%d  base=%d(%.3f)  faulty=%d(%.3f)  %s\n",
                B.image_name.c_str(), sefi_name(mode),
                B.ground_truth_class, B.baseline_class, B.baseline_prob,
                IR.top1, IR.top1_prob,
                RES.correctly_classified ? "CORRECT" : "WRONG");
    return true;
}

// =============================================================================
// CSV OUTPUT
// =============================================================================
static void write_results_csv(const vector<RunResultSEFI>& results,
                               const string& out_dir, const string& mode_name) {
    string path = out_dir + "/results_" + mode_name + ".csv";
    ofstream f(path);
    if (!f) { fprintf(stderr, "[CSV] Cannot write %s\n", path.c_str()); return; }

    f << "image_name,sefi_mode,target,transient,"
         "ground_truth_class,ground_truth_name,"
         "baseline_class,baseline_name,baseline_prob,"
         "faulty_top1,faulty_top1_name,faulty_top1_prob,"
         "faulty_top2,faulty_top2_name,faulty_top2_prob,"
         "faulty_top3,faulty_top3_name,faulty_top3_prob,"
         "correctly_classified,prob_drop,"
         "bytes_corrupted,bits_corrupted,fault_phys_addr,"
         "msefi_ran,msefi_recovered,recovery_class,recovery_prob,recovery_action,"
         "crash\n";

    for (auto& R : results) {
        auto q = [](const string& s) { return "\"" + s + "\""; };
        f << q(R.image_name) << "," << q(R.mode_name) << ","
          << q(targetName(R.target)) << "," << (R.transient_mode ? 1 : 0) << ","
          << R.ground_truth_class << "," << q(R.ground_truth_name) << ","
          << R.baseline_class     << "," << q(R.baseline_name) << ","
          << fixed << setprecision(6) << R.baseline_prob << ","
          << R.faulty_class[0] << "," << q(R.faulty_name[0]) << "," << R.faulty_prob[0] << ","
          << R.faulty_class[1] << "," << q(R.faulty_name[1]) << "," << R.faulty_prob[1] << ","
          << R.faulty_class[2] << "," << q(R.faulty_name[2]) << "," << R.faulty_prob[2] << ","
          << (R.correctly_classified ? 1 : 0) << "," << R.prob_drop << ","
          << R.bytes_corrupted << "," << R.bits_corrupted << ","
          << "0x" << hex << R.fault_phys_addr << dec << ","
          << (R.msefi_ran       ? 1 : 0) << "," << (R.msefi_recovered ? 1 : 0) << ","
          << R.recovery_class << "," << R.recovery_prob << "," << q(R.recovery_action) << ","
          << (R.crash ? 1 : 0) << "\n";
    }
    printf("[CSV] Saved: %s\n", path.c_str());
}

static void write_accuracy_csv(const AccuracyRow& row, const string& out_dir) {
    string path = out_dir + "/accuracy_summary.csv";
    ofstream f(path);
    if (!f) { fprintf(stderr, "[CSV] Cannot write %s\n", path.c_str()); return; }
    f << "sefi_mode,total_images,"
         "baseline_correct,baseline_accuracy_pct,"
         "faulty_correct,faulty_wrong,faulty_accuracy_pct,"
         "msefi_recovered,recovery_accuracy_pct\n";
    f << row.mode_name << "," << row.total_images << ","
      << row.baseline_correct << "," << fixed << setprecision(2) << row.baseline_pct << ","
      << row.faulty_correct << "," << row.faulty_wrong << "," << row.faulty_pct << ","
      << row.recovered_correct << "," << row.recovery_pct << "\n";
    printf("[CSV] Saved: %s\n", path.c_str());
}

// =============================================================================
// INTERACTIVE MENU
// =============================================================================
static void print_sefi_menu() {
    printf("\n");
    printf("╔══════════════════════════════════════════════════════════════════════╗\n");
    printf("║      SEFI Fault Injection Simulator — Mode Selection                ║\n");
    printf("║  Ref: Guertin DDR4 NEPP FY24 Final Report, Sec.4 Tables 1 & 2      ║\n");
    printf("╠══════════════════════════════════════════════════════════════════════╣\n");
    printf("║  Spatial SEFI (DDR4 /dev/mem direct injection):                     ║\n");
    printf("║   1.  SEFI-row            Full DDR4 row (~8 KB) corrupted           ║\n");
    printf("║   2.  Transient SEFI-row  Row clears on row re-read                 ║\n");
    printf("║   3.  SEFI-column(band)   Stripe: same column-offset across rows    ║\n");
    printf("║   4.  Transient SEFI-col  Stripe clears on bank/row switch          ║\n");
    printf("║   5.  SEFI-block(other)   Contiguous block corrupted (config size)  ║\n");
    printf("║   6.  Transient SEFI-blk  Block clears on row/bank switch           ║\n");
    printf("╠══════════════════════════════════════════════════════════════════════╣\n");
    printf("║  Management SEFI (tests DPU/runner recovery):                       ║\n");
    printf("║   7.  MSEFI – DDR reset req          DDR partial reinit fixes it    ║\n");
    printf("║   8.  MSEFI – ctrl reset req         Controller-only reset fixes    ║\n");
    printf("║   9.  MSEFI – ctrl+DDR reset req     Both ctrl AND DDR reset needed ║\n");
    printf("║  10.  MSEFI – DDR power cycle req    DDR must be power-cycled       ║\n");
    printf("║  11.  MSEFI – full power cycle req   Both DDR and ctrl power-cycled ║\n");
    printf("╠══════════════════════════════════════════════════════════════════════╣\n");
    printf("║  NOT SIMULATABLE:                                                   ║\n");
    printf("║   -   fixable SEFI-row/col/block  Needs DDR4 MRS cmd (no OS API)   ║\n");
    printf("║   -   SEL (Single Event Latchup)  Physical latchup — destroys board ║\n");
    printf("╚══════════════════════════════════════════════════════════════════════╝\n\n");
}

static SEFIMode select_sefi_mode() {
    print_sefi_menu();
    printf("Enter mode number [1-11]: ");
    fflush(stdout);
    string line; getline(cin, line);
    int choice = 5;
    try { choice = stoi(line); } catch (...) {}
    switch (choice) {
        case  1: return SEFIMode::SEFI_ROW;
        case  2: return SEFIMode::TRANSIENT_SEFI_ROW;
        case  3: return SEFIMode::SEFI_COLUMN;
        case  4: return SEFIMode::TRANSIENT_SEFI_COLUMN;
        case  5: return SEFIMode::SEFI_BLOCK;
        case  6: return SEFIMode::TRANSIENT_SEFI_BLOCK;
        case  7: return SEFIMode::MSEFI_DDR_RESET;
        case  8: return SEFIMode::MSEFI_CTRL_RESET;
        case  9: return SEFIMode::MSEFI_CTRL_DDR_RESET;
        case 10: return SEFIMode::MSEFI_DDR_POWER_CYCLE;
        case 11: return SEFIMode::MSEFI_FULL_POWER_CYCLE;
        default:
            printf("[Menu] Unknown choice %d — defaulting to SEFI-block\n", choice);
            return SEFIMode::SEFI_BLOCK;
    }
}

static FaultTarget parse_target(const string& s) {
    string lo = s; transform(lo.begin(), lo.end(), lo.begin(), ::tolower);
    if (lo == "weights")                                              return FaultTarget::WEIGHTS;
    if (lo == "instructions")                                         return FaultTarget::INSTRUCTIONS;
    if (lo == "input_tensor" || lo == "feature_maps" || lo == "input") return FaultTarget::INPUT_TENSOR;
    if (lo == "buffers"      || lo == "output")                       return FaultTarget::BUFFERS;
    if (lo == "all")                                                  return FaultTarget::ALL;
    fprintf(stderr, "[Config] Unknown target '%s', using weights\n", s.c_str());
    return FaultTarget::WEIGHTS;
}

// =============================================================================
// MAIN
// =============================================================================
int main(int argc, char* argv[]) {
    if (argc < 2) {
        printf("Usage: %s <model.xmodel> [target] [-v]\n", argv[0]);
        printf("  target: weights | instructions | input_tensor | buffers | all\n");
        return -1;
    }

    mt19937 rng(static_cast<uint32_t>(time(nullptr)) ^ (uint32_t)getpid());

    g_devmem_fd = open("/dev/mem", O_RDWR | O_SYNC);
    if (g_devmem_fd < 0) {
        fprintf(stderr, "[DDR4] Cannot open /dev/mem — must run as root.\n");
        return -1;
    }
    printf("[DDR4] /dev/mem opened (fd=%d). DDR4 direct injection enabled.\n", g_devmem_fd);
    printf("[DDR4] DPU S_AXI control base: 0x%lX\n\n", DPU_CTRL_BASE);

    SimConfig cfg;
    cfg.model_path = argv[1];
    if (argc >= 3) cfg.target = parse_target(argv[2]);
    cfg.verbose = (argc >= 4 && string(argv[3]) == "-v");

    cfg.mode = select_sefi_mode();
    const char* mname = sefi_name(cfg.mode);
    bool        msefi = is_msefi(cfg.mode);

    if (cfg.mode == SEFIMode::SEFI_COLUMN || cfg.mode == SEFIMode::TRANSIENT_SEFI_COLUMN) {
        printf("Enter column band width in bytes [default 8 = one DDR4 burst beat]: ");
        fflush(stdout);
        string line; getline(cin, line);
        try { cfg.col_width = (size_t)stoul(line); } catch (...) { cfg.col_width = DDR4_COL_DEFAULT; }
        cfg.col_width = max((size_t)1, cfg.col_width);
    }

    if (cfg.mode == SEFIMode::SEFI_BLOCK          || cfg.mode == SEFIMode::TRANSIENT_SEFI_BLOCK ||
        cfg.mode == SEFIMode::MSEFI_DDR_RESET      || cfg.mode == SEFIMode::MSEFI_CTRL_RESET    ||
        cfg.mode == SEFIMode::MSEFI_CTRL_DDR_RESET || cfg.mode == SEFIMode::MSEFI_DDR_POWER_CYCLE ||
        cfg.mode == SEFIMode::MSEFI_FULL_POWER_CYCLE) {
        printf("Enter corruption block size in bytes [default 4096]: ");
        fflush(stdout);
        string line; getline(cin, line);
        try { cfg.block_size = (size_t)stoul(line); } catch (...) { cfg.block_size = 4096; }
        cfg.block_size = max((size_t)8, cfg.block_size);
    }

    printf("\nEnter image folder path [default ./train_subset]: ");
    fflush(stdout);
    { string line; getline(cin, line); cfg.val_folder = line.empty() ? "./train_subset" : line; }

    if (argc < 3) {
        printf("Target [weights/instructions/input_tensor/buffers/all, default weights]: ");
        fflush(stdout);
        string line; getline(cin, line);
        if (!line.empty()) cfg.target = parse_target(line);
    }

    printf("\n[Config] SEFI mode    = %s\n", mname);
    printf("[Config] target       = %s\n",   targetName(cfg.target).c_str());
    printf("[Config] image folder = %s\n",   cfg.val_folder.c_str());
    printf("[Config] results in   = ./FaultResults/sefi_results/%s/<target>/\n",
           sefi_folder_name(cfg.mode).c_str());
    if (cfg.mode == SEFIMode::SEFI_COLUMN || cfg.mode == SEFIMode::TRANSIENT_SEFI_COLUMN)
        printf("[Config] col_width    = %zu B\n", cfg.col_width);
    printf("[Config] block_size   = %zu B\n", cfg.block_size);

    mkdirp("./FaultResults/sefi_results/" + sefi_folder_name(cfg.mode));
    string logpath = "./FaultResults/sefi_results/" + sefi_folder_name(cfg.mode) + "/sefi_sim.log";
    g_logfp = fopen(logpath.c_str(), "w");
    if (!g_logfp) fprintf(stderr, "[Warn] Cannot open log %s\n", logpath.c_str());

    vector<string> kinds;
    LoadWords(wordsPath + "words.txt", kinds);
    map<string, int> synset_to_idx = LoadSynsets(wordsPath + "synset.txt");
    if (synset_to_idx.empty()) { fprintf(stderr, "[Error] synset.txt missing\n"); return -1; }

    vector<ImageEntry> entries;
    ListImagesWithGroundTruth(cfg.val_folder, synset_to_idx, entries);
    if (entries.empty()) {
        fprintf(stderr, "[Error] No images in %s\n", cfg.val_folder.c_str()); return -1;
    }
    printf("[Setup] %zu images found\n", entries.size());

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

    // ── Baseline pass ─────────────────────────────────────────────────────────
    printf("[Baseline] Running clean model on %zu images...\n", entries.size());
    vector<BaselineResult> baselines;
    vector<vector<int8_t>> imgBufs;
    baselines.reserve(entries.size());
    imgBufs.reserve(entries.size());

    for (size_t i = 0; i < entries.size(); i++) {
        printf("\r[Baseline] %zu / %zu  ", i + 1, entries.size()); fflush(stdout);
        baselines.push_back(compute_baseline(runner, entries[i], kinds));
        vector<int8_t> buf(inSz, 0);
        Mat raw = imread(entries[i].path);
        if (!raw.empty()) preprocess_image(raw, buf.data(), inH, inW, in_sc);
        imgBufs.push_back(move(buf));
    }
    printf("\r[Baseline] Done.                    \n");

    // Cache DDR4 addresses (control registers populated by VART after baseline)
    cache_instr_address();
    cache_weights_address();
    cache_input_address();

    int base_correct = 0, base_total = 0;
    for (auto& B : baselines) {
        if (!B.valid) continue;
        base_total++;
        if (B.baseline_class == B.ground_truth_class) base_correct++;
    }
    float base_pct = base_total > 0 ? 100.f * base_correct / base_total : 0.f;
    printf("[Baseline] Clean accuracy: %d/%d = %.2f%%\n", base_correct, base_total, base_pct);
    sim_log("[Baseline] Accuracy: %d/%d = %.2f%%\n", base_correct, base_total, base_pct);

    // ── Target list: ALL excludes instructions (only weights, input_tensor, buffers)
    static const FaultTarget ALL_TARGETS[] = {
        FaultTarget::WEIGHTS,
        FaultTarget::INPUT_TENSOR,
        FaultTarget::BUFFERS
    };
    vector<FaultTarget> targets_to_run;
    if (cfg.target == FaultTarget::ALL)
        targets_to_run.assign(ALL_TARGETS, ALL_TARGETS + 3);
    else
        targets_to_run.push_back(cfg.target);

    // ── Per-target injection loop ─────────────────────────────────────────────
    for (FaultTarget cur_target : targets_to_run) {
        string tname   = targetName(cur_target);
        string out_dir = prepare_output_dir(cfg.mode, tname);

        SimConfig tcfg = cfg;
        tcfg.target    = cur_target;

        printf("\n[Run] SEFI mode: %s  target: %s  images: %zu\n",
               mname, tname.c_str(), entries.size());
        sim_log("\n──── target=%s ────\n", tname.c_str());

        vector<RunResultSEFI> results;
        results.reserve(entries.size());
        int total_correct = 0, total_recovered = 0, img_total = 0;

        for (size_t img_idx = 0; img_idx < entries.size(); img_idx++) {
            const BaselineResult& B = baselines[img_idx];
            if (!B.valid) continue;

            printf("\r  [%zu/%zu] %s  target=%s  ",
                   img_idx + 1, entries.size(), B.image_name.c_str(), tname.c_str());
            fflush(stdout);

            RunResultSEFI R;
            bool ok = perform_sefi_run(runner, subgraph[0], imgBufs[img_idx],
                                       B, kinds, tcfg, rng, R);

            if (!ok && !R.crash) {
                sim_log("[Main] Hard crash img=%s — recreating runner\n", B.image_name.c_str());
                auto nr = recreate_runner(subgraph[0]);
                if (nr) runner = nr.release();
            }

            for (int i = 0; i < 3; i++)
                if (R.faulty_name[i].empty() && R.faulty_class[i] >= 0 &&
                    R.faulty_class[i] < (int)kinds.size())
                    R.faulty_name[i] = kinds[R.faulty_class[i]];

            if (R.correctly_classified) total_correct++;
            if (R.msefi_recovered)      total_recovered++;
            img_total++;
            results.push_back(R);
        }
        printf("\r  Done %d images                              \n", img_total);

        write_results_csv(results, out_dir, mname);

        float faulty_pct   = img_total > 0 ? 100.f * total_correct   / img_total : 0.f;
        float recovery_pct = img_total > 0 ? 100.f * total_recovered / img_total : 0.f;

        AccuracyRow acc;
        acc.mode_name         = mname;
        acc.total_images      = img_total;
        acc.baseline_correct  = base_correct;   acc.baseline_pct   = base_pct;
        acc.faulty_correct    = total_correct;  acc.faulty_wrong   = img_total - total_correct;
        acc.faulty_pct        = faulty_pct;
        acc.recovered_correct = total_recovered; acc.recovery_pct  = recovery_pct;
        write_accuracy_csv(acc, out_dir);

        printf("  [Summary] target=%-14s  baseline=%.2f%%  faulty=%.2f%%",
               tname.c_str(), base_pct, faulty_pct);
        if (msefi) printf("  recovery=%.2f%%", recovery_pct);
        printf("\n");
        sim_log("[Summary] target=%s  baseline=%.2f%%  faulty=%.2f%%  recovery=%.2f%%\n",
                tname.c_str(), base_pct, faulty_pct, recovery_pct);
    }

    // ── Final summary ─────────────────────────────────────────────────────────
    printf("\n╔══════════════════════════════════════════════════════╗\n");
    printf("║  SEFI Simulation Done — %-28s║\n", mname);
    printf("╠══════════════════════════════════════════════════════╣\n");
    printf("║  Baseline (clean model): %d/%d = %.2f%%\n", base_correct, base_total, base_pct);
    printf("╠══════════════════════════════════════════════════════╣\n");
    printf("║  Results saved in:\n");
    printf("║    ./FaultResults/sefi_results/%s/\n", sefi_folder_name(cfg.mode).c_str());
    for (FaultTarget t : targets_to_run)
        printf("║      %s/\n", targetName(t).c_str());
    printf("║  Each folder: results_%s.csv\n", mname);
    printf("║               accuracy_summary.csv\n");
    printf("╚══════════════════════════════════════════════════════╝\n");
    printf("\nTo generate plots, run on host:\n");
    printf("  python3 ./FaultResults/sefi_results/plot_all.py\n");

    if (g_logfp) fclose(g_logfp);
    if (g_devmem_fd >= 0) close(g_devmem_fd);
    return 0;
}
