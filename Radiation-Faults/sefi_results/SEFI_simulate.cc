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
 *                           (per-region — instructions excluded physically)
 *   4.  Transient SEFI-col  Column stripe, clears on bank/row switch
 *   5.  SEFI-block(other)   Contiguous block corruption (configurable size)
 *   6.  Transient SEFI-blk  Block clears on row/bank switch
 *
 *  MSEFI (Massive SEFI) — NOT included in this simulator. MSEFI is a total
 *    DDR4 communication breakdown (AXI bus hang, Linux panic). Bit-flip
 *    injection cannot simulate it. See SEFIMode enum comment below for the
 *    correct ZCU104 approach (attack PS DDRC at 0xFD070000 or DDR4 RESET_n GPIO).
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
// SEFI MODE ENUM  (spatial only — MSEFI removed, see note below)
//
// ── WHY MSEFI IS NOT HERE ────────────────────────────────────────────────────
// MSEFI = Massive SEFI (historically "Million error SEFI")
// Source: Guertin, NEPP DDR4 FY24 Final Report, Section 2 (Acronyms), 2025.
//
// MSEFI is a TOTAL DDR4 communication breakdown — one radiation particle hits
// the DDR4 device's internal state machine or PHY, causing it to stop
// responding to ALL commands. The AXI bus hangs. The CPU throws a Data Abort.
// Linux panics. The DDR4 device physically stops talking.
//
// WHY BIT-FLIP INJECTION CANNOT SIMULATE MSEFI:
//   Bit flips corrupt data payload. But during any bit-flip injection in this
//   code, the DDR4 device is still successfully receiving Read/Write commands
//   and successfully returning responses. The communication channel is intact.
//   That is the OPPOSITE of what MSEFI represents.
//
// HOW TO ACTUALLY SIMULATE MSEFI ON ZCU104:
//   Method 1 (Controller Reset req):
//     Write to PS DDRC soft-reset register at 0xFD070000 + offset mid-inference.
//     Forces AXI timeout. Tests OS recovery without DDR4 power cycle.
//   Method 2 (DDR+Controller Reset req):
//     Scramble DDRC DQS timing registers (0xFD070xxx) mid-inference.
//     Causes DQS alignment failure → DDR4 stops responding.
//   Method 3 (Power Cycle req):
//     If DDR4 RESET_n is routed to PL GPIO in block design, assert it low
//     mid-inference. Physically holds DDR4 in hardware reset — most authentic.
//
// CONCLUSION: The "MSEFI" modes previously in this code were large SEFI-block
// injections (data corruption, not communication failure) and were removed to
// avoid misrepresenting the Guertin taxonomy.
// =============================================================================
enum class SEFIMode {
    SEFI_ROW,
    TRANSIENT_SEFI_ROW,
    SEFI_COLUMN,
    TRANSIENT_SEFI_COLUMN,
    SEFI_BLOCK,
    TRANSIENT_SEFI_BLOCK,
};

static const char* sefi_name(SEFIMode m) {
    switch (m) {
        case SEFIMode::SEFI_ROW:               return "SEFI-row";
        case SEFIMode::TRANSIENT_SEFI_ROW:     return "transient-SEFI-row";
        case SEFIMode::SEFI_COLUMN:            return "SEFI-column";
        case SEFIMode::TRANSIENT_SEFI_COLUMN:  return "transient-SEFI-col";
        case SEFIMode::SEFI_BLOCK:             return "SEFI-block";
        case SEFIMode::TRANSIENT_SEFI_BLOCK:   return "transient-SEFI-blk";
    }
    return "unknown";
}

static string sefi_folder_name(SEFIMode m) {
    switch (m) {
        case SEFIMode::SEFI_ROW:               return "01. SEFI-row";
        case SEFIMode::TRANSIENT_SEFI_ROW:     return "02. transient-SEFI-row";
        case SEFIMode::SEFI_COLUMN:            return "03. SEFI-column";
        case SEFIMode::TRANSIENT_SEFI_COLUMN:  return "04. transient-SEFI-col";
        case SEFIMode::SEFI_BLOCK:             return "05. SEFI-block";
        case SEFIMode::TRANSIENT_SEFI_BLOCK:   return "06. transient-SEFI-blk";
    }
    return "00. unknown";
}

static bool is_transient(SEFIMode m) {
    return m == SEFIMode::TRANSIENT_SEFI_ROW   ||
           m == SEFIMode::TRANSIENT_SEFI_COLUMN ||
           m == SEFIMode::TRANSIENT_SEFI_BLOCK;
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

// Writes a formatted message to both stdout and the log file simultaneously.
// fmt: printf-style format string.
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
// Creates a full directory path recursively (like mkdir -p).
// path: full directory path to create.
static void mkdirp(const string& path) {
    string tmp = path;
    for (size_t i = 1; i < tmp.size(); i++) {
        if (tmp[i] == '/') { tmp[i] = '\0'; mkdir(tmp.c_str(), 0755); tmp[i] = '/'; }
    }
    mkdir(tmp.c_str(), 0755);
}

// Deletes all regular files inside a directory (non-recursive, leaves subdirs).
// path: directory to clear.
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

// Builds the output directory path for a given mode+target, creates it, and clears old results.
// results_folder: user-chosen top-level folder name (e.g. "sefi_results").
// Returns the full path e.g. ./FaultResults/<results_folder>/01. SEFI-row/weights/
static string prepare_output_dir(const string& results_folder, SEFIMode mode, const string& tname) {
    string tdir = "./FaultResults/" + results_folder + "/" + sefi_folder_name(mode) + "/" + tname;
    mkdirp(tdir); clear_dir(tdir);
    printf("[Dir] Output: %s\n", tdir.c_str());
    return tdir;
}

// =============================================================================
// DDR4 CONTROL REGISTER ACCESS
// =============================================================================
// Reads one 64-bit DPU control register by mmapping the DPU S_AXI base (0x80000000).
// off_lo/off_hi: byte offsets of the low and high 32-bit halves of the register.
// VART writes DDR4 physical addresses into these registers during runner initialisation.
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

// Reads dpu_instr_addr register and converts PFN to physical: g_instr_phys = reg_val << 12.
// Instructions use PFN encoding (unlike the 1:1 physical mapping of weights/input/output).
static void cache_instr_address() {
    uint64_t pfn = read_ctrl_reg64(OFF_INSTR_LO, OFF_INSTR_HI);
    g_instr_phys = pfn << 12;
    if (g_instr_phys)
        sim_log("[DDR4] Instructions: reg_val=0x%lX  phys=0x%016lX  size=%zu B\n",
                pfn, g_instr_phys, DDR4_INSTR_SIZE);
    else
        fprintf(stderr, "[DDR4] WARNING: instr_phys=0 after baseline\n");
}

// Reads dpu_base0_addr (1:1 physical) into g_weights_phys. Called once after baseline.
static void cache_weights_address() {
    g_weights_phys = read_ctrl_reg64(OFF_BASE0_LO, OFF_BASE0_HI);
    if (g_weights_phys)
        sim_log("[DDR4] Weights:      phys=0x%016lX  size=%zu B\n",
                g_weights_phys, DDR4_WEIGHT_SIZE);
    else
        fprintf(stderr, "[DDR4] WARNING: weights_phys=0 after baseline\n");
}

// Reads dpu_base2_addr (1:1 physical) into g_input_phys.
// Pixel data starts at g_input_phys + DDR4_INPUT_HDR (VART prepends a 2080-byte header).
static void cache_input_address() {
    // dpu_base2_addr: 1:1 physical mapping. Pixel data at +DDR4_INPUT_HDR (2080 B).
    g_input_phys = read_ctrl_reg64(OFF_BASE2_LO, OFF_BASE2_HI);
    if (g_input_phys)
        sim_log("[DDR4] Input tensor: phys=0x%016lX  total=%zu B  pixel_start=+%zu B\n",
                g_input_phys, DDR4_INPUT_SIZE, DDR4_INPUT_HDR);
    else
        fprintf(stderr, "[DDR4] WARNING: input_phys=0 after baseline\n");
}

// Reads dpu_base3_addr fresh on every call (not cached).
// Returns the current physical address of the 1008-byte output buffer.
static uint64_t read_output_address() {
    return read_ctrl_reg64(OFF_BASE3_LO, OFF_BASE3_HI);
}

// =============================================================================
// REGION FLIP RECORD
// =============================================================================
struct RegionFlip {
    uint64_t phys_base    = 0;
    size_t   region_size  = 0;
    vector<pair<size_t, uint8_t>> restores;   // (byte offset from phys_base, orig)
    size_t bytes_affected = 0;
    size_t bits_corrupted = 0;
};

// For cross-region column injection: stores absolute physical addresses
struct AbsFlip { uint64_t phys_addr; uint8_t before; };

// Restores bytes recorded as AbsFlip (absolute physical address + original value).
// Used only for SEFI-column which spans disjoint regions and cannot use a shared phys_base.
// One mmap per byte — more expensive than restore_region() but required for cross-region restore.
static void restore_abs_flips(const vector<AbsFlip>& flips) {
    if (g_devmem_fd < 0 || flips.empty()) return;
    for (auto& f : flips) {
        uint64_t pg  = f.phys_addr & ~(uint64_t)4095;
        size_t   adj = (size_t)(f.phys_addr - pg);
        void* m = mmap(NULL, adj + 1, PROT_READ | PROT_WRITE, MAP_SHARED,
                       g_devmem_fd, (off_t)pg);
        if (m == MAP_FAILED) continue;
        ((uint8_t*)m)[adj] = f.before;
        munmap(m, adj + 1);
    }
}

// Maps a physical DDR4 region read-write via /dev/mem, handling 4096-byte page alignment.
// phys_base: target physical address; sz: bytes to map.
// pg_base/adj/map_sz: OUT — caller must munmap(returned_ptr - adj, map_sz) when done.
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

// Writes back all original byte values saved in a RegionFlip to their DDR4 addresses.
// Called for transient modes after inference completes, and for permanent modes at end of run.
static void restore_region(const RegionFlip& rf) {
    if (g_devmem_fd < 0 || rf.restores.empty()) return;
    uint64_t pg; size_t adj, msz;
    uint8_t* base = region_map_rw(rf.phys_base, rf.region_size, pg, adj, msz);
    if (!base) return;
    for (auto& [off, orig] : rf.restores) base[off] = orig;
    munmap(base - adj, msz);
}

// =============================================================================
// WEIGHT SNAPSHOT (Option B — fresh-state-per-image experimental design)
// =============================================================================
// Sequence per image: 1) INITIALIZE (restore weights from backup)
//                     2) FAULT INJECTION  3) INFERENCE
//                     4) ANALYSIS  5) LOGGING  6) repeat from 1.
// The snapshot is taken ONCE at startup right after cache_weights_address(),
// when DDR4 weights are guaranteed clean (only baseline inferences have run).
// restore_weights_from_backup() is then called at the start of EVERY faulty
// run so each image is an independent single-event measurement, even in
// permanent SEFI modes where the injection itself is never reverted.
static vector<uint8_t> g_weights_backup;

// Copies the full DDR4 weight region into a CPU-side backup buffer.
// Must be called once, after cache_weights_address(), before any injection.
static bool snapshot_weights() {
    if (g_devmem_fd < 0 || g_weights_phys == 0) return false;
    uint64_t pg; size_t adj, msz;
    uint8_t* base = region_map_rw(g_weights_phys, DDR4_WEIGHT_SIZE, pg, adj, msz);
    if (!base) return false;
    g_weights_backup.resize(DDR4_WEIGHT_SIZE);
    memcpy(g_weights_backup.data(), base, DDR4_WEIGHT_SIZE);
    munmap(base - adj, msz);
    printf("[Snapshot] Weight region backed up: %zu bytes from 0x%016lX\n",
           (size_t)DDR4_WEIGHT_SIZE, g_weights_phys);
    sim_log("[Snapshot] Weight backup taken: %zu B @ 0x%016lX\n",
            (size_t)DDR4_WEIGHT_SIZE, g_weights_phys);
    return true;
}

// Restores the full DDR4 weight region from the CPU backup. Called as the
// INITIALIZATION step (step 1) before every faulty run. ~25 MB memcpy ≈ 50 ms.
static bool restore_weights_from_backup() {
    if (g_devmem_fd < 0 || g_weights_phys == 0 || g_weights_backup.empty())
        return false;
    uint64_t pg; size_t adj, msz;
    uint8_t* base = region_map_rw(g_weights_phys, DDR4_WEIGHT_SIZE, pg, adj, msz);
    if (!base) return false;
    memcpy(base, g_weights_backup.data(), DDR4_WEIGHT_SIZE);
    munmap(base - adj, msz);
    return true;
}

// =============================================================================
// CROSS-REGION HELPERS
// =============================================================================

// Check which known DPU DDR4 regions [phys, phys+sz) overlaps and return
// a comma-separated string, e.g. "weights,input_tensor,buffers".
// Also logs it. Used for SEFI-row and SEFI-column which cross region boundaries.
// Checks which DPU DDR4 regions a physical address range [phys, phys+sz) overlaps.
// Returns a comma-separated string e.g. "weights,input_tensor". Also logs the result.
static string compute_regions_covered(uint64_t phys, size_t sz,
                                       uint64_t out_phys, const char* tag) {
    struct { const char* name; uint64_t base; size_t size; } known[] = {
        { "weights",      g_weights_phys, DDR4_WEIGHT_SIZE  },
        { "input_tensor", g_input_phys,   DDR4_INPUT_SIZE   },
        { "output",       out_phys,       DDR4_OUTPUT_SIZE  },
        { "instructions", g_instr_phys,   DDR4_INSTR_SIZE   },
    };
    string covered;
    uint64_t end = phys + sz;
    for (auto& r : known) {
        if (r.base == 0 || r.size == 0) continue;
        uint64_t rend = r.base + r.size;
        if (phys < rend && end > r.base) {  // overlap
            if (!covered.empty()) covered += ",";
            covered += r.name;
        }
    }
    sim_log("  [%s] regions_covered: [%s]  phys=0x%016lX  size=%zu B\n",
            tag, covered.c_str(), phys, sz);
    return covered;
}

// SEFI-row cross-region: pick a random 8KB-aligned row anchored inside
// target region, but map and corrupt the full DDR4_ROW_BYTES regardless
// of whether it extends past the target's end. This reflects the physical
// reality that a row-activation event affects the entire 8KB physical row.
// SEFI-row injection: picks a random 8192-byte DDR4-row-aligned block anchored inside
// [anchor_phys, anchor_phys+anchor_sz), then maps and XORs all 8192 bytes with random
// non-zero masks regardless of region boundary (physically correct — row events cross regions).
// anchor_phys/anchor_sz: defines where to anchor the row; rng: random engine.
static RegionFlip inject_sefi_row_cross(uint64_t anchor_phys, size_t anchor_sz,
                                         mt19937& rng, bool verbose, const char* tag) {
    RegionFlip rf;
    if (g_devmem_fd < 0 || anchor_phys == 0) return rf;

    // Pick a row-aligned address that starts within [anchor_phys, anchor_phys+anchor_sz)
    // Clamp so at least 1 byte of the row is inside the target
    size_t n_positions = max((size_t)1, anchor_sz / DDR4_ROW_BYTES + 1);
    uniform_int_distribution<size_t> rdist(0, n_positions - 1);
    uint64_t row_base = (anchor_phys & ~(uint64_t)(DDR4_ROW_BYTES - 1))
                        + rdist(rng) * DDR4_ROW_BYTES;

    rf.phys_base   = row_base;
    rf.region_size = DDR4_ROW_BYTES;  // always full row, may cross region boundary

    uint64_t pg; size_t adj, msz;
    uint8_t* base = region_map_rw(row_base, DDR4_ROW_BYTES, pg, adj, msz);
    if (!base) return rf;

    uniform_int_distribution<uint8_t> maskd(1, 255);
    size_t bits = 0;
    for (size_t i = 0; i < DDR4_ROW_BYTES; i++) {
        uint8_t xmask = maskd(rng);
        uint8_t orig  = base[i];
        base[i] ^= xmask;
        rf.restores.push_back({i, orig});
        bits += (size_t)__builtin_popcount(xmask);
    }
    rf.bytes_affected = DDR4_ROW_BYTES;
    rf.bits_corrupted = bits;

    sim_log("  [%s] SEFI-row(cross-region)  row_phys=0x%016lX  size=%zu B  bits~%zu\n",
            tag, row_base, DDR4_ROW_BYTES, bits);
    (void)verbose;
    munmap(base - adj, msz);
    return rf;
}

// inject_sefi_column_cross: stripe the same column offset across all DPU DDR4
// data regions. Each region is mapped INDEPENDENTLY via separate mmap calls.
// NEVER computes a contiguous span — instructions live in the gaps and would
// be hit by a single span mmap (→ kernel panic, as confirmed in testing).
// Restores stored as AbsFlip (absolute physical address) since offsets span
// multiple disjoint regions and cannot share a single phys_base.
//
// PARTIAL ROW HANDLING (important):
//   Most regions are not exact multiples of DDR4_ROW_BYTES (8 KB).
//   e.g. weights = 25,726,976 B = 3140 complete rows + 4096 B remainder.
//   The column stripe hits the partial row if col_start < remainder_bytes.
//   This is counted separately as a "partial row" contribution.
//
//   Small regions (< DDR4_ROW_BYTES, e.g. output buffer = 1008 B) also get
//   hit if col_start < region_size. The buffer occupies part of one physical
//   DDR4 row — if the column offset falls within the buffer, those bytes
//   get corrupted. Expected bytes hit ≈ col_width × (region_size/8192).
//
// Per-region byte counts returned via out_bytes_* for CSV logging.
// SEFI-column injection: picks ONE random column offset and applies it to weights, input,
// and output regions via SEPARATE mmaps (never a contiguous span — instructions live in
// the physical gap and would be hit by a single mmap, causing kernel panic).
// Handles partial rows at the end of each region. Per-region byte counts returned via out_bytes_*.
// col_width: bytes to corrupt per row (default 8); abs_flips/out_bytes_*: OUT parameters.
static RegionFlip inject_sefi_column_cross(size_t col_width,
                                            mt19937& rng, bool verbose,
                                            const char* tag,
                                            vector<AbsFlip>& abs_flips,
                                            size_t& out_bytes_weights,
                                            size_t& out_bytes_input,
                                            size_t& out_bytes_output) {
    RegionFlip rf;
    abs_flips.clear();
    out_bytes_weights = out_bytes_input = out_bytes_output = 0;
    if (g_devmem_fd < 0) return rf;

    uint64_t out_phys = read_output_address();

    struct Seg { uint64_t base; size_t size; const char* name; size_t* counter; };
    vector<Seg> segs;
    if (g_weights_phys) segs.push_back({g_weights_phys, DDR4_WEIGHT_SIZE, "weights", &out_bytes_weights});
    if (g_input_phys)   segs.push_back({g_input_phys,   DDR4_INPUT_SIZE,  "input",   &out_bytes_input});
    if (out_phys)       segs.push_back({out_phys,        DDR4_OUTPUT_SIZE, "output",  &out_bytes_output});
    if (segs.empty()) return rf;

    if (col_width == 0 || col_width > DDR4_ROW_BYTES) col_width = DDR4_COL_DEFAULT;

    // Pick ONE column offset — same for all regions (same physical column address)
    uniform_int_distribution<size_t>  cdist(0, DDR4_ROW_BYTES - col_width);
    uniform_int_distribution<uint8_t> maskd(1, 255);
    size_t col_start = cdist(rng);
    size_t total_bits = 0;

    for (auto& seg : segs) {
        // How many complete 8 KB rows fit in this region?
        size_t n_complete_rows = seg.size / DDR4_ROW_BYTES;
        // Remainder bytes after last complete row (the "partial row")
        size_t remainder      = seg.size % DDR4_ROW_BYTES;
        // Total logical rows to iterate (complete + 1 partial if remainder > 0)
        size_t n_rows_total   = n_complete_rows + (remainder > 0 ? 1 : 0);

        if (n_rows_total == 0) continue;  // empty region, skip

        uint64_t pg; size_t adj, msz;
        uint8_t* base = region_map_rw(seg.base, seg.size, pg, adj, msz);
        if (!base) continue;

        size_t region_bytes = 0;
        for (size_t row = 0; row < n_rows_total; row++) {
            size_t row_start_off = row * DDR4_ROW_BYTES;
            // How many bytes are available in this row within the region?
            size_t row_avail = (row < n_complete_rows)
                               ? DDR4_ROW_BYTES          // complete row: all 8192 B
                               : remainder;              // partial row: only remainder B

            // Column stripe hits this row only if col_start < row_avail
            if (col_start >= row_avail) continue;

            // How many bytes of col_width fit within the available bytes?
            size_t bytes_in_row = min(col_width, row_avail - col_start);
            size_t byte_off     = row_start_off + col_start;

            for (size_t c = 0; c < bytes_in_row; c++) {
                uint8_t xmask = maskd(rng);
                uint8_t orig  = base[byte_off + c];
                base[byte_off + c] ^= xmask;
                abs_flips.push_back({seg.base + byte_off + c, orig});
                total_bits  += (size_t)__builtin_popcount(xmask);
                region_bytes++;
            }
        }
        *seg.counter = region_bytes;
        munmap(base - adj, msz);

        sim_log("  [%s] SEFI-col '%s'  col_off=%zu  col_w=%zu  "
                "complete_rows=%zu  partial=%zu B  bytes_hit=%zu\n",
                tag, seg.name, col_start, col_width,
                n_complete_rows, remainder, region_bytes);
    }

    rf.phys_base      = segs[0].base;
    rf.region_size    = segs[0].size;
    rf.bytes_affected = out_bytes_weights + out_bytes_input + out_bytes_output;
    rf.bits_corrupted = total_bits;

    sim_log("  [%s] SEFI-col TOTAL  bytes=%zu (weights=%zu input=%zu output=%zu)  bits~%zu\n",
            tag, rf.bytes_affected, out_bytes_weights, out_bytes_input,
            out_bytes_output, total_bits);
    (void)verbose;
    return rf;
}
// ─────────────────────────────────────────────────────
// Paper (Table 1): "A SEFI where a set of bits are corrupted (e.g. by having
//   a refresh operation start without the read system ready to obtain the data)."
// Method: Pick a random DDR4-row-aligned block (8 KB) in the target region.
//   XOR every byte with a random non-zero mask → full-row corruption signature.
//   Transient: restore all bytes after inference (clears on row re-read).
// SEFI-block injection: picks a uniform-random start offset inside the target region,
// then XORs block_sz contiguous bytes with random non-zero masks. Region-confined.
// phys_base/region_sz: target DDR4 region; block_sz: bytes to corrupt (clamped to region_sz).
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
// Resizes image to DPU input size, subtracts ImageNet BGR mean [104,107,123],
// and quantises to INT8 using the VART input scale factor.
// src: OpenCV BGR image; dst: INT8 output buffer (inH×inW×3 bytes); scale: from get_input_scale().
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

// Converts DPU raw INT8 output logits to float probabilities via softmax on the CPU.
// d: INT8 logits from DPU; sz: number of classes (1000); scale: from get_output_scale().
static void CPUCalcSoftmax(const int8_t* d, int sz, float* out, float scale) {
    double sum = 0.0;
    for (int i = 0; i < sz; i++) { out[i] = expf((float)d[i] * scale); sum += out[i]; }
    for (int i = 0; i < sz; i++) out[i] /= (float)sum;
}

// Returns indices of the k highest-probability classes, sorted descending.
// p: probability array (length sz); k: how many top results to return.
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

// Runs one DPU forward pass: wraps imgBuf/fcBuf in VART TensorBuffers, calls
// execute_async + wait, computes softmax on CPU, and returns top-K results.
// runner: VART runner; imgBuf: INT8 input; fcBuf: INT8 output buffer (pre-allocated).
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

// hardware_reset_dpu() and recreate_runner() removed.
// These were used only by MSEFI modes which have been removed.
// See SEFIMode enum comment for how to properly simulate MSEFI on ZCU104.

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
    string   regions_covered;
    // Per-region byte breakdown (relevant for cross-region modes)
    size_t   bytes_weights  = 0;
    size_t   bytes_input    = 0;
    size_t   bytes_output   = 0;   // e.g. "weights,input_tensor,buffers" — for cross-region modes

    // MSEFI fields removed — MSEFI cannot be simulated via bit-flip injection.
    // See SEFIMode enum comment for correct MSEFI simulation approaches.

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
    // Baseline metrics
    int    baseline_correct;   float baseline_pct;
    // Faulty metrics
    int    faulty_correct;     int   faulty_wrong;    float faulty_pct;
    // Unused (MSEFI removed)
    int    recovered_correct = 0; float recovery_pct = 0.f;
};

// =============================================================================
// DATA LOADING
// =============================================================================
// Reads synset.txt (one WordNet synset ID per line) and maps each synset to its 0-based class index.
// path: path to synset.txt.
static map<string, int> LoadSynsets(const string& path) {
    map<string, int> m;
    ifstream f(path);
    if (!f) { fprintf(stderr, "[Warn] synset.txt not found: %s\n", path.c_str()); return m; }
    string line; int idx = 0;
    while (getline(f, line)) { if (!line.empty()) m[line] = idx; idx++; }
    return m;
}

// Walks val_dir/<synset>/*.jpg structure, builds a sorted list of ImageEntry records
// with ground-truth class labels from synset_to_idx.
// val_dir: root image folder; synset_to_idx: from LoadSynsets(); entries: OUT vector.
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

// Reads words.txt (one human-readable class description per line) into a vector.
// Line index equals class index. kinds: OUT vector.
static void LoadWords(const string& path, vector<string>& kinds) {
    kinds.clear(); ifstream f(path);
    if (!f) { fprintf(stderr, "[Error] Cannot open: %s\n", path.c_str()); exit(1); }
    string line; while (getline(f, line)) kinds.push_back(line);
}

// =============================================================================
// BASELINE
// =============================================================================
// Runs a clean (no injection) inference on one image and records the result as the baseline.
// Called once per image before any fault injection. entry: image path + ground_truth; kinds: class names.
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
// SINGLE SEFI FAULTY RUN DISPATCHER
// =============================================================================
// Main dispatcher for one image's fault injection experiment.
// Resolves the DDR4 target region, calls the appropriate inject_* function,
// runs faulty inference via run_inference(), handles transient restore, and fills RES.
// Returns true on success; false on hard crash (DPU exception).
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

    // ── STEP 1: INITIALIZATION (Option B) ─────────────────────────────────────
    // Restore DDR4 weights to the clean snapshot so this image is an
    // INDEPENDENT single-event experiment. Without this, permanent SEFI modes
    // accumulate corruption across images (confirmed in sefi_results_02:
    // BLOCK/weights degraded 7/10 correct → 0/10, faulty_top1 collapsed to
    // class 456 by image 41). Runs for every image regardless of target —
    // a previous weights-target run may have left corruption behind.
    if (!restore_weights_from_backup()) {
        sim_log("[Init] WARNING: weight restore failed for %s\n",
                B.image_name.c_str());
    }

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
        // rgsz is capped to the actual allocated DDR4 region minus the header
        // so the inject_* functions and the memcpy below cannot overrun into
        // other DPU regions or OS memory.
        phys = g_input_phys + DDR4_INPUT_HDR;
        rgsz = min((size_t)inSz, DDR4_INPUT_SIZE - DDR4_INPUT_HDR);
    }

    if (!use_output && phys == 0) { RES.crash = true; return false; }

    vector<int8_t> img(imgBuf);
    RegionFlip     rf;

    // SEFI-column is the ONE mode where a single physical event spans MULTIPLE
    // regions simultaneously (one DDR4 column line crosses weights, input, AND
    // output in the same access). For row/block modes the buffers target stays
    // output-only, since a row/block event is spatially confined to wherever
    // it lands. Routing column's buffers target through the same cross-region
    // call used for weights/input_tensor targets ensures weights+input are
    // ALSO corrupted here — previously this path injected into output ONLY,
    // which under-reported a real column SEFI's actual blast radius.
    bool column_mode = (mode == SEFIMode::SEFI_COLUMN ||
                        mode == SEFIMode::TRANSIENT_SEFI_COLUMN);

    if (use_output && column_mode) {
        // Inject weights + input_tensor + output together BEFORE inference,
        // exactly like the weights/input_tensor target dispatch below does.
        vector<AbsFlip> col_abs_flips;
        rf = inject_sefi_column_cross(cfg.col_width, rng, cfg.verbose,
                                      sefi_name(mode), col_abs_flips,
                                      RES.bytes_weights, RES.bytes_input,
                                      RES.bytes_output);
        string cov;
        if (RES.bytes_weights) cov += "weights";
        if (RES.bytes_input)   cov += (cov.empty() ? "" : ",") + string("input_tensor");
        if (RES.bytes_output)  cov += (cov.empty() ? "" : ",") + string("output");
        if (cov.empty()) cov = "none(column missed all regions)";
        cov += "(instructions excluded)";
        RES.regions_covered = cov;
        RES.fault_phys_addr = col_abs_flips.empty() ? 0 : col_abs_flips[0].phys_addr;
        RES.bytes_corrupted = rf.bytes_affected;
        RES.bits_corrupted  = rf.bits_corrupted;

        vector<int8_t> fcBuf(outSz, 0);
        auto IR0 = run_inference(runner, img.data(), inSz, inH, inW,
                                 fcBuf.data(), outSz, out_sc, inT[0], outT[0]);
        if (!IR0.ok) {
            if (RES.transient_mode) restore_abs_flips(col_abs_flips);
            RES.crash = true; return false;
        }
        if (RES.transient_mode) restore_abs_flips(col_abs_flips);

        // Output buffer (already corrupted above if the column hit it) — read
        // post-inference so the corrupted bytes (if any) are reflected in fcBuf.
        uint64_t out_phys2 = read_output_address();
        if (out_phys2 != 0) {
            uint64_t pg3 = out_phys2 & ~(uint64_t)4095;
            size_t   adj3 = (size_t)(out_phys2 - pg3);
            size_t   msz3 = DDR4_OUTPUT_SIZE + adj3;
            void* dm3 = mmap(NULL, msz3, PROT_READ, MAP_SHARED, g_devmem_fd, (off_t)pg3);
            if (dm3 != MAP_FAILED) {
                memcpy(fcBuf.data(),
                       reinterpret_cast<int8_t*>((uint8_t*)dm3 + adj3),
                       min((size_t)outSz, DDR4_OUTPUT_SIZE));
                munmap(dm3, msz3);
            }
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

    if (use_output) {
        vector<int8_t> fcBuf(outSz, 0);
        auto IR0 = run_inference(runner, img.data(), inSz, inH, inW,
                                 fcBuf.data(), outSz, out_sc, inT[0], outT[0]);
        if (!IR0.ok) { RES.crash = true; return false; }

        uint64_t out_phys = read_output_address();
        RegionFlip rfo;
        if (out_phys != 0) {
            // Mode-aware output buffer injection:
            //   SEFI-row:    the 1008 B buffer sits inside ONE 8 KB DDR4 row, so a
            //                row event corrupts the whole buffer → full 1008 B. ✓
            //   SEFI-column: stripe = col_width bytes at a random column offset.
            //                The offset is drawn from [0, 8192-col_width); it hits
            //                the buffer only if col_start < 1008. A miss (0 bytes)
            //                is a VALID physical outcome and is recorded as such.
            //   SEFI-block:  contiguous block clamped to buffer size (as before).
            if (mode == SEFIMode::SEFI_COLUMN ||
                mode == SEFIMode::TRANSIENT_SEFI_COLUMN) {
                size_t cw = cfg.col_width ? cfg.col_width : DDR4_COL_DEFAULT;
                cw = min(cw, DDR4_ROW_BYTES);
                uniform_int_distribution<size_t> cdist(0, DDR4_ROW_BYTES - cw);
                size_t col_start = cdist(rng);

                rfo.phys_base   = out_phys;
                rfo.region_size = DDR4_OUTPUT_SIZE;
                if (col_start < DDR4_OUTPUT_SIZE) {
                    size_t n = min(cw, DDR4_OUTPUT_SIZE - col_start);
                    uint64_t pg2; size_t adj2, msz2;
                    uint8_t* b2 = region_map_rw(out_phys, DDR4_OUTPUT_SIZE,
                                                pg2, adj2, msz2);
                    if (b2) {
                        uniform_int_distribution<uint8_t> maskd(1, 255);
                        size_t bits = 0;
                        for (size_t c = 0; c < n; c++) {
                            uint8_t xm   = maskd(rng);
                            uint8_t orig = b2[col_start + c];
                            b2[col_start + c] ^= xm;
                            rfo.restores.push_back({col_start + c, orig});
                            bits += (size_t)__builtin_popcount(xm);
                        }
                        munmap(b2 - adj2, msz2);
                        rfo.bytes_affected = n;
                        rfo.bits_corrupted = bits;
                        sim_log("  [buffers_post] SEFI-col stripe  col_off=%zu "
                                "col_w=%zu  bytes_hit=%zu\n", col_start, cw, n);
                    }
                } else {
                    sim_log("  [buffers_post] SEFI-col stripe MISSED buffer "
                            "(col_off=%zu >= %zu)  bytes_hit=0\n",
                            col_start, (size_t)DDR4_OUTPUT_SIZE);
                }
            } else {
                // SEFI-row → full buffer (inside one physical row).
                // SEFI-block → block_size clamped to buffer.
                size_t blk = (mode == SEFIMode::SEFI_ROW ||
                              mode == SEFIMode::TRANSIENT_SEFI_ROW)
                             ? DDR4_OUTPUT_SIZE
                             : min(cfg.block_size, DDR4_OUTPUT_SIZE);
                rfo = inject_sefi_block(out_phys, DDR4_OUTPUT_SIZE,
                                        blk, rng, cfg.verbose, "buffers_post");
            }

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
            RES.fault_phys_addr = out_phys + (rfo.restores.empty() ? 0 : rfo.restores[0].first);
            // Only claim the region if bytes actually landed there
            RES.regions_covered = rfo.bytes_affected > 0 ? "output" : "none(column missed buffer)";
            RES.bytes_output    = rfo.bytes_affected;
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

    // ── Spatial injection ──────────────────────────────────────────────────────
    uint64_t out_phys_for_log = read_output_address();
    vector<AbsFlip> col_abs_flips;  // only used by SEFI-column
    switch (mode) {
        case SEFIMode::SEFI_ROW:
        case SEFIMode::TRANSIENT_SEFI_ROW: {
            rf = inject_sefi_row_cross(phys, rgsz, rng, cfg.verbose, sefi_name(mode));
            RES.regions_covered = compute_regions_covered(
                rf.phys_base, rf.region_size, out_phys_for_log, sefi_name(mode));
            // fault_phys_addr = the actual row base that was injected (random per image)
            RES.fault_phys_addr = rf.phys_base;

            // Per-region byte breakdown: overlap of [row_base, row_base+8192)
            // with each known region (row events physically cross boundaries).
            auto overlap = [](uint64_t a0, uint64_t a1,
                              uint64_t b0, uint64_t b1) -> size_t {
                uint64_t lo = max(a0, b0), hi = min(a1, b1);
                return hi > lo ? (size_t)(hi - lo) : 0;
            };
            uint64_t r0 = rf.phys_base, r1 = rf.phys_base + rf.region_size;
            if (g_weights_phys)
                RES.bytes_weights = overlap(r0, r1, g_weights_phys,
                                            g_weights_phys + DDR4_WEIGHT_SIZE);
            if (g_input_phys)
                RES.bytes_input   = overlap(r0, r1, g_input_phys,
                                            g_input_phys + DDR4_INPUT_SIZE);
            if (out_phys_for_log)
                RES.bytes_output  = overlap(r0, r1, out_phys_for_log,
                                            out_phys_for_log + DDR4_OUTPUT_SIZE);
            break;
        }

        case SEFIMode::SEFI_COLUMN:
        case SEFIMode::TRANSIENT_SEFI_COLUMN: {
            rf = inject_sefi_column_cross(cfg.col_width, rng, cfg.verbose,
                                          sefi_name(mode), col_abs_flips,
                                          RES.bytes_weights, RES.bytes_input,
                                          RES.bytes_output);
            // Only list regions where bytes actually landed (output is missed
            // whenever col_start >= 1008 — probability ~88% with 8 KB rows).
            string cov;
            if (RES.bytes_weights) cov += "weights";
            if (RES.bytes_input)   cov += (cov.empty() ? "" : ",") + string("input_tensor");
            if (RES.bytes_output)  cov += (cov.empty() ? "" : ",") + string("output");
            cov += "(instructions excluded)";
            RES.regions_covered = cov;
            // fault_phys_addr = physical address of first byte actually flipped
            RES.fault_phys_addr = col_abs_flips.empty() ? 0 : col_abs_flips[0].phys_addr;
            break;
        }

        case SEFIMode::SEFI_BLOCK:
        case SEFIMode::TRANSIENT_SEFI_BLOCK:
            rf = inject_sefi_block(phys, rgsz, cfg.block_size,
                                   rng, cfg.verbose, sefi_name(mode));
            RES.regions_covered  = targetName(eff_target);
            RES.fault_phys_addr  = phys + (rf.restores.empty() ? 0 : rf.restores[0].first);

            // Block is region-confined by design: all bytes land in eff_target.
            if (eff_target == FaultTarget::WEIGHTS)
                RES.bytes_weights = rf.bytes_affected;
            else if (eff_target == FaultTarget::INPUT_TENSOR)
                RES.bytes_input   = rf.bytes_affected;
            break;

        default: break;
    }

    RES.bytes_corrupted = rf.bytes_affected;
    RES.bits_corrupted  = rf.bits_corrupted;

    // ── Run faulty inference ──────────────────────────────────────────────────
    vector<int8_t> fcBuf(outSz, 0);
    auto IR = run_inference(runner, img.data(), inSz, inH, inW,
                            fcBuf.data(), outSz, out_sc, inT[0], outT[0]);

    // ── Transient: restore DDR4 bytes before capturing result ─────────────────
    if (RES.transient_mode) {
        if (!col_abs_flips.empty())
            restore_abs_flips(col_abs_flips);   // column uses abs addresses
        else
            restore_region(rf);                 // all other modes use region offset
    }

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
// METRICS COMPUTATION
// =============================================================================

// =============================================================================
// CSV OUTPUT
// =============================================================================
// Writes the full per-image fault injection results to a CSV file.
// Includes fault metadata (bytes/bits/regions) and per-region byte counts.
// results: all RunResultSEFI for this target; out_dir: destination folder; mode_name: for filename.
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
         "bytes_corrupted,bits_corrupted,fault_phys_addr,regions_covered,"
         "bytes_in_weights,bytes_in_input,bytes_in_output,"
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
          << q(R.regions_covered) << ","
          << R.bytes_weights << "," << R.bytes_input << "," << R.bytes_output << ","
          << (R.crash ? 1 : 0) << "\n";
    }
    printf("[CSV] Saved: %s\n", path.c_str());
}

// Writes a single-row accuracy summary CSV for one mode+target combination.
static void write_accuracy_csv(const AccuracyRow& row, const string& out_dir) {
    string path = out_dir + "/accuracy_summary.csv";
    ofstream f(path);
    if (!f) { fprintf(stderr, "[CSV] Cannot write %s\n", path.c_str()); return; }
    f << "sefi_mode,total_images,"
         "baseline_correct,baseline_accuracy_pct,"
         "faulty_correct,faulty_wrong,faulty_accuracy_pct\n";
    f << row.mode_name << "," << row.total_images << ","
      << row.baseline_correct << ","
      << fixed << setprecision(4) << row.baseline_pct << ","
      << row.faulty_correct << "," << row.faulty_wrong << ","
      << row.faulty_pct << "\n";
    printf("[CSV] Saved: %s\n", path.c_str());
}

// =============================================================================
// INTERACTIVE MENU
// =============================================================================
// Prints the interactive SEFI mode selection menu to stdout (modes 1-6 only).
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
    printf("║  NOT SIMULATABLE via bit-flip injection:                            ║\n");
    printf("║   -   MSEFI (Massive/Million error SEFI) — total DDR4 communication ║\n");
    printf("║       breakdown. Requires attacking PS DDRC registers (0xFD070000)  ║\n");
    printf("║       or asserting DDR4 RESET_n GPIO pin mid-inference.             ║\n");
    printf("║   -   fixable SEFI-row/col/block — needs DDR4 MRS command injection ║\n");
    printf("║   -   SEL — physical latchup, destroys board                        ║\n");
    printf("╚══════════════════════════════════════════════════════════════════════╝\n\n");
}

// Reads the user's integer mode choice (1-6) from stdin and returns the corresponding SEFIMode.
// Defaults to SEFI_BLOCK on invalid input.
static SEFIMode select_sefi_mode() {
    print_sefi_menu();
    printf("Enter mode number [1-6]: ");
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
        default:
            printf("[Menu] Unknown choice %d — defaulting to SEFI-block\n", choice);
            return SEFIMode::SEFI_BLOCK;
    }
}

// Parses a user-typed target string to a FaultTarget enum value.
// Accepts aliases: "feature_maps" → INPUT_TENSOR, "output" → BUFFERS.
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

    // Results folder prompt
    printf("Results folder name [default sefi_results]: ");
    fflush(stdout);
    string results_folder;
    getline(cin, results_folder);
    if (results_folder.empty()) results_folder = "sefi_results";
    const char* mname = sefi_name(cfg.mode);

    if (cfg.mode == SEFIMode::SEFI_COLUMN || cfg.mode == SEFIMode::TRANSIENT_SEFI_COLUMN) {
        printf("Enter column band width in bytes [default 8 = one DDR4 burst beat]: ");
        fflush(stdout);
        string line; getline(cin, line);
        try { cfg.col_width = (size_t)stoul(line); } catch (...) { cfg.col_width = DDR4_COL_DEFAULT; }
        cfg.col_width = max((size_t)1, cfg.col_width);
    }

    if (cfg.mode == SEFIMode::SEFI_BLOCK || cfg.mode == SEFIMode::TRANSIENT_SEFI_BLOCK) {
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
    printf("[Config] results in   = ./FaultResults/%s/%s/<target>/\n",
           results_folder.c_str(), sefi_folder_name(cfg.mode).c_str());
    if (cfg.mode == SEFIMode::SEFI_COLUMN || cfg.mode == SEFIMode::TRANSIENT_SEFI_COLUMN)
        printf("[Config] col_width    = %zu B\n", cfg.col_width);
    printf("[Config] block_size   = %zu B\n", cfg.block_size);

    mkdirp("./FaultResults/" + results_folder + "/" + sefi_folder_name(cfg.mode));
    string logpath = "./FaultResults/" + results_folder + "/" + sefi_folder_name(cfg.mode) + "/sefi_sim.log";
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

    // Option B: snapshot clean weights ONCE. Restored before every faulty run
    // (step 1 of the per-image sequence) so every image starts from identical
    // clean state, even in permanent SEFI modes.
    if (!snapshot_weights()) {
        fprintf(stderr, "[Error] Weight snapshot failed — cannot guarantee "
                        "fresh state per image. Aborting.\n");
        return -1;
    }

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
        string out_dir = prepare_output_dir(results_folder, cfg.mode, tname);

        SimConfig tcfg = cfg;
        tcfg.target    = cur_target;

        printf("\n[Run] SEFI mode: %s  target: %s  images: %zu\n",
               mname, tname.c_str(), entries.size());
        sim_log("\n──── target=%s ────\n", tname.c_str());

        vector<RunResultSEFI> results;
        results.reserve(entries.size());
        int total_correct = 0, img_total = 0;

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
                sim_log("[Main] Hard crash img=%s — runner kept (no recreate available)\n",
                        B.image_name.c_str());
            }

            for (int i = 0; i < 3; i++)
                if (R.faulty_name[i].empty() && R.faulty_class[i] >= 0 &&
                    R.faulty_class[i] < (int)kinds.size())
                    R.faulty_name[i] = kinds[R.faulty_class[i]];

            if (R.correctly_classified) total_correct++;
            img_total++;
            results.push_back(R);
        }
        printf("\r  Done %d images                              \n", img_total);

        write_results_csv(results, out_dir, mname);

        float faulty_pct = img_total > 0 ? 100.f * total_correct / img_total : 0.f;

        AccuracyRow acc;
        acc.mode_name        = mname;
        acc.total_images     = img_total;
        acc.baseline_correct = base_correct;  acc.baseline_pct = base_pct;
        acc.faulty_correct   = total_correct; acc.faulty_wrong  = img_total - total_correct;
        acc.faulty_pct       = faulty_pct;
        write_accuracy_csv(acc, out_dir);

        printf("  [Summary] target=%-14s  baseline=%.2f%%  faulty=%.2f%%\n",
               tname.c_str(), base_pct, faulty_pct);
        sim_log("[Summary] target=%s  baseline=%.2f%%  faulty=%.2f%%\n",
                tname.c_str(), base_pct, faulty_pct);
    }

    // ── Final summary ─────────────────────────────────────────────────────────
    printf("\n╔══════════════════════════════════════════════════════╗\n");
    printf("║  SEFI Simulation Done — %-28s║\n", mname);
    printf("╠══════════════════════════════════════════════════════╣\n");
    printf("║  Baseline (clean model): %d/%d = %.2f%%\n", base_correct, base_total, base_pct);
    printf("╠══════════════════════════════════════════════════════╣\n");
    printf("║  Results saved in:\n");
    printf("║    ./FaultResults/%s/%s/\n", results_folder.c_str(), sefi_folder_name(cfg.mode).c_str());
    for (FaultTarget t : targets_to_run)
        printf("║      %s/\n", targetName(t).c_str());
    printf("║  Each folder: results_%s.csv\n", mname);
    printf("║               accuracy_summary.csv\n");
    printf("╚══════════════════════════════════════════════════════╝\n");
    printf("\nTo generate plots, run on host:\n");
    printf("  python3 ./FaultResults/sefi_plot.py\n");

    if (g_logfp) fclose(g_logfp);
    if (g_devmem_fd >= 0) close(g_devmem_fd);
    return 0;
}
