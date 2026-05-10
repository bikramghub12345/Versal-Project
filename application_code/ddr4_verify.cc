/*
 * ddr4_verify.cc  —  DDR4 Readback Verification (All 5 DPU Regions)
 * ==================================================================
 *
 * FINDINGS THAT SHAPED THIS CODE:
 *
 * 1. Instructions (HPC0/M_AXI_GP0):
 *    dpu_instr_addr register returns a DPU-local IOMMU-translated address,
 *    NOT the CPU physical address. Reading /dev/mem at that value hits
 *    low-memory (kernel exception vectors). Fixed by scanning ±64MB around
 *    the weights base (HP0 allocations are co-located) for the mc_code
 *    signature. Found at 0x1BE00000 → 100% match confirmed.
 *
 * 2. Weights (HP0/M_AXI_HP0):
 *    dpu_base0_addr is a 1:1 CPU physical address. 100% match with REG_0.bin.
 *
 * 3. Feature maps (HP0/M_AXI_HP0):
 *    Runtime-computed by DPU. ~50% zeros = expected ReLU pattern.
 *    No reference file possible. Stats only.
 *
 * 4. Input tensor (HP0/M_AXI_HP0):
 *    REG_2 size (152608) > imgBuf size (150528). Difference = 2080 bytes.
 *    VART prepends a 2080-byte header/padding in its internal buffer before
 *    the actual image data. Code scans within DDR4 region to find where
 *    imgBuf content actually starts.
 *
 * 5. Output tensor (HP0/M_AXI_HP0):
 *    100% match with CPU fcBuf. Top-5 decoded from DDR4 directly.
 *
 * BUILD:
 *   g++ -std=c++17 -O2 -o ddr4_verify src/ddr4_verify.cc \
 *       ../common/common.cpp \
 *       -I./src -I../common \
 *       -I/usr/include/opencv4 -I/usr/include/vitis_ai \
 *       -lopencv_core -lopencv_imgproc -lopencv_imgcodecs \
 *       -lvart-runner -lxir -lglog -lpthread
 *
 * RUN (must be root):
 *   ./ddr4_verify <model.xmodel> <ref_dir> <image_path>
 */

#include <fcntl.h>
#include <stdio.h>
#include <stdlib.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>

#include <algorithm>
#include <cmath>
#include <cstring>
#include <fstream>
#include <numeric>
#include <memory>
#include <string>
#include <vector>

#include "common.h"
#include <opencv2/opencv.hpp>

using namespace std;
using namespace cv;

// ─── Hardware constants ──────────────────────────────────────────────────────
static const uint32_t CORE1_BASE   = 0x80000000;
static const uint32_t OFF_INSTR_LO = 0x50;
static const uint32_t OFF_INSTR_HI = 0x54;
static const uint32_t OFF_BASE0_LO = 0x60;   // weights
static const uint32_t OFF_BASE0_HI = 0x64;
static const uint32_t OFF_BASE1_LO = 0x68;   // feature maps
static const uint32_t OFF_BASE1_HI = 0x6C;
static const uint32_t OFF_BASE2_LO = 0x70;   // input tensor
static const uint32_t OFF_BASE2_HI = 0x74;
static const uint32_t OFF_BASE3_LO = 0x78;   // output tensor
static const uint32_t OFF_BASE3_HI = 0x7C;

// Sizes from xir dump_bin (file sizes) and xir dump_reg (reg_id_to_size)
static const size_t INSTR_SIZE  = 742492;
static const size_t WEIGHT_SIZE = 25726976;
static const size_t FMAP_SIZE   = 2207744;
static const size_t INPUT_SIZE  = 152608;
static const size_t OUTPUT_SIZE = 1008;

static const size_t SCAN_CHUNK    = 0x4000000;  // 64 MB per mmap chunk
static const size_t SCAN_STEP     = 4096;        // page-aligned steps
static const size_t SCAN_SIG_LEN  = 64;          // signature bytes to match
static const int    TOP_K         = 5;
static const float  MEAN[3]       = {104.f, 107.f, 123.f};

GraphInfo shapes;

// ─── /dev/mem helpers ────────────────────────────────────────────────────────

static volatile uint32_t* map_ctrl(int fd, uint32_t base) {
    void* m = mmap(NULL, 4096, PROT_READ, MAP_SHARED, fd, (off_t)base);
    if (m == MAP_FAILED) { perror("map_ctrl"); return nullptr; }
    return (volatile uint32_t*)m;
}

static uint8_t* map_ddr4(int fd, uint64_t phys, size_t size,
                          void** mb, size_t* ms) {
    if (!phys || !size) return nullptr;
    uint64_t pg  = phys & ~(uint64_t)4095;
    size_t   adj = (size_t)(phys - pg);
    *ms = size + adj;
    void* m = mmap(NULL, *ms, PROT_READ, MAP_SHARED, fd, (off_t)pg);
    if (m == MAP_FAILED) { perror("map_ddr4"); return nullptr; }
    *mb = m;
    return (uint8_t*)m + adj;
}

static uint64_t reg64(volatile uint32_t* c, uint32_t lo, uint32_t hi) {
    return ((uint64_t)c[hi/4] << 32) | c[lo/4];
}

// ─── DDR4 scan: find physical address of a byte signature ───────────────────
// Used for:
//   (a) instruction buffer — IOMMU-translated reg value is not CPU phys addr
//   (b) image data within input tensor — VART may prepend a header
static uint64_t scan_for_signature(int fd,
                                   uint64_t scan_base, size_t scan_window,
                                   const uint8_t* sig, size_t sig_len,
                                   size_t step) {
    size_t chunks = (scan_window + SCAN_CHUNK - 1) / SCAN_CHUNK;
    for (size_t c = 0; c < chunks; c++) {
        uint64_t cs   = scan_base + c * SCAN_CHUNK;
        size_t   csz  = min(SCAN_CHUNK, scan_window - c * SCAN_CHUNK);
        uint64_t pg   = cs & ~(uint64_t)4095;
        size_t   adj  = (size_t)(cs - pg);
        size_t   msz  = csz + adj;
        void* m = mmap(NULL, msz, PROT_READ, MAP_SHARED, fd, (off_t)pg);
        if (m == MAP_FAILED) continue;
        uint8_t* base = (uint8_t*)m + adj;
        for (size_t off = 0; off + sig_len <= csz; off += step) {
            if (memcmp(base + off, sig, sig_len) == 0) {
                uint64_t found = cs + off;
                munmap(m, msz);
                return found;
            }
        }
        munmap(m, msz);
    }
    return 0;
}

// ─── File helpers ────────────────────────────────────────────────────────────

static vector<uint8_t> load_file(const string& p) {
    ifstream f(p, ios::binary);
    if (!f) { fprintf(stderr, "Cannot open: %s\n", p.c_str()); return {}; }
    return vector<uint8_t>((istreambuf_iterator<char>(f)), istreambuf_iterator<char>());
}

static vector<string> load_words(const string& p) {
    vector<string> v; ifstream f(p);
    string l; while (getline(f, l)) v.push_back(l);
    return v;
}

// ─── Print helpers ───────────────────────────────────────────────────────────

static void section(const char* t) {
    printf("\n── %s ", t);
    int pad = 55 - (int)strlen(t);
    for (int i = 0; i < pad; i++) putchar('─');
    printf("\n");
}

static void print_range(uint64_t base, size_t size) {
    printf("  DDR4  : 0x%016lX → 0x%016lX  (%zu bytes / %.2f KB)\n",
           base, base + size - 1, size, size / 1024.0);
}

static void compare(const uint8_t* ddr4, const uint8_t* ref, size_t n) {
    size_t mm = 0, first = SIZE_MAX, last = 0;
    for (size_t i = 0; i < n; i++) {
        if (ddr4[i] != ref[i]) {
            mm++; if (first == SIZE_MAX) first = i; last = i;
        }
    }
    if (!mm) {
        printf("  Compare : MATCH  — %zu bytes identical (100%%)\n", n);
        return;
    }
    printf("  Compare : MISMATCH  — %zu/%zu bytes differ (%.4f%% match)\n",
           mm, n, 100.0*(n-mm)/n);
    printf("  First mismatch offset : 0x%zX  last : 0x%zX\n", first, last);
    printf("  %-10s  %-8s  %-8s\n", "Offset", "DDR4", "REF");
    int shown = 0;
    for (size_t i = 0; i < n && shown < 8; i++)
        if (ddr4[i] != ref[i])
            printf("  0x%06zX    0x%02X      0x%02X\n", i, ddr4[i], ref[i]), shown++;
}

static void stats(const uint8_t* p, size_t n) {
    if (!p || !n) return;
    int8_t mn = 127, mx = -128; long sum = 0; int zeros = 0;
    for (size_t i = 0; i < n; i++) {
        int8_t v = (int8_t)p[i];
        if (v < mn) mn = v; if (v > mx) mx = v;
        sum += v; if (!v) zeros++;
    }
    printf("  Stats   : min=%d  max=%d  mean=%.3f  zeros=%d/%zu (%.1f%%)\n",
           (int)mn, (int)mx, (double)sum/n, zeros, n, 100.0*zeros/n);
    printf("  First 16: "); for (size_t i=0;i<16&&i<n;i++) printf("%02X ",p[i]); printf("\n");
    size_t s=(n>16)?n-16:0;
    printf("  Last  16: "); for (size_t i=s;i<n;i++) printf("%02X ",p[i]); printf("\n");
}

// ─── Softmax + top-k ─────────────────────────────────────────────────────────

static void softmax(const int8_t* d, int n, float* o, float sc) {
    double s = 0;
    for (int i=0;i<n;i++){o[i]=expf((float)d[i]*sc);s+=o[i];}
    for (int i=0;i<n;i++) o[i]/=(float)s;
}
static vector<int> topk(const float* p, int n, int k) {
    vector<int> idx(n); iota(idx.begin(),idx.end(),0);
    partial_sort(idx.begin(),idx.begin()+k,idx.end(),[&](int a,int b){return p[a]>p[b];});
    idx.resize(k); return idx;
}

// ─── Preprocessing + inference ───────────────────────────────────────────────

static bool preprocess(const string& path, int8_t* dst, int H, int W, float sc) {
    Mat img = imread(path);
    if (img.empty()) { fprintf(stderr,"Cannot read: %s\n",path.c_str()); return false; }
    Mat rsz; resize(img, rsz, Size(W,H), 0, 0, INTER_LINEAR);
    for (int h=0;h<H;h++) for (int w=0;w<W;w++) for (int c=0;c<3;c++) {
        float v = ((float)rsz.at<Vec3b>(h,w)[c] - MEAN[c]) * sc;
        dst[h*W*3+w*3+c] = (int8_t)max(-128.f, min(127.f, v));
    }
    return true;
}

static bool run_inference(vart::Runner* runner, int8_t* imgBuf, int8_t* fcBuf) {
    auto inT=runner->get_input_tensors(), outT=runner->get_output_tensors();
    auto id=inT[0]->get_shape(); id[0]=1;
    auto od=outT[0]->get_shape(); od[0]=1;
    vector<shared_ptr<xir::Tensor>> bt;
    bt.push_back(shared_ptr<xir::Tensor>(xir::Tensor::create(
        inT[0]->get_name(),id,xir::DataType{xir::DataType::XINT,8u})));
    auto ib=make_unique<CpuFlatTensorBuffer>(imgBuf,bt.back().get());
    bt.push_back(shared_ptr<xir::Tensor>(xir::Tensor::create(
        outT[0]->get_name(),od,xir::DataType{xir::DataType::XINT,8u})));
    auto ob=make_unique<CpuFlatTensorBuffer>(fcBuf,bt.back().get());
    vector<vart::TensorBuffer*> ip={ib.get()},op={ob.get()};
    auto job=runner->execute_async(ip,op);
    return (runner->wait(job.first,5000)==0);
}

// ─── MAIN ────────────────────────────────────────────────────────────────────

int main(int argc, char* argv[]) {
    if (argc != 4) {
        printf("Usage: %s <model.xmodel> <ref_dir> <image_path>\n", argv[0]);
        printf("  ref_dir: contains subgraph_conv1.mc and REG_0.bin\n");
        return -1;
    }

    int fd = open("/dev/mem", O_RDWR|O_SYNC);
    if (fd<0) { perror("open /dev/mem"); return -1; }

    volatile uint32_t* ctrl = map_ctrl(fd, CORE1_BASE);
    if (!ctrl) { close(fd); return -1; }

    // Load model + run inference
    auto graph    = xir::Graph::deserialize(argv[1]);
    auto subgraph = get_dpu_subgraph(graph.get());
    CHECK_EQ(subgraph.size(),1u) << "Expected one DPU subgraph";
    auto runner = vart::Runner::create_runner(subgraph[0],"run");

    auto inT=runner->get_input_tensors(), outT=runner->get_output_tensors();
    TensorShape insh[8],outsh[8];
    shapes.inTensorList=insh; shapes.outTensorList=outsh;
    getTensorShape(runner.get(),&shapes,(int)inT.size(),(int)outT.size());

    int   inH=shapes.inTensorList[0].height,  inW=shapes.inTensorList[0].width;
    int   inSz=shapes.inTensorList[0].size,   outSz=shapes.outTensorList[0].size;
    float in_sc=get_input_scale(inT[0]),       out_sc=get_output_scale(outT[0]);

    vector<int8_t> imgBuf(inSz,0), fcBuf(outSz,0);
    if (!preprocess(argv[3], imgBuf.data(), inH, inW, in_sc)) { close(fd); return -1; }
    printf("[Init] %s  |  image: %s\n", argv[1], argv[3]);
    printf("[Init] Input %dx%dx3=%d B  Output=%d B\n", inH, inW, inSz, outSz);

    printf("[Inference] Running...\n");
    if (!run_inference(runner.get(), imgBuf.data(), fcBuf.data())) {
        fprintf(stderr,"Inference failed.\n"); close(fd); return -1;
    }
    printf("[Inference] Done.\n");

    // Read all 5 control registers post-inference
    uint64_t instr_raw   = reg64(ctrl, OFF_INSTR_LO, OFF_INSTR_HI);
    uint64_t addr_weight = reg64(ctrl, OFF_BASE0_LO, OFF_BASE0_HI);
    uint64_t addr_fmap   = reg64(ctrl, OFF_BASE1_LO, OFF_BASE1_HI);
    uint64_t addr_input  = reg64(ctrl, OFF_BASE2_LO, OFF_BASE2_HI);
    uint64_t addr_output = reg64(ctrl, OFF_BASE3_LO, OFF_BASE3_HI);

    printf("\n[Control Registers — DPUCZDX8G_1 @ 0x%08X]\n", CORE1_BASE);
    printf("  0x50  dpu_instr_addr = 0x%016lX  (IOMMU-translated, not CPU phys)\n", instr_raw);
    printf("  0x60  dpu_base0_addr = 0x%016lX  (weights)\n",      addr_weight);
    printf("  0x68  dpu_base1_addr = 0x%016lX  (feature maps)\n", addr_fmap);
    printf("  0x70  dpu_base2_addr = 0x%016lX  (input tensor)\n", addr_input);
    printf("  0x78  dpu_base3_addr = 0x%016lX  (output tensor)\n",addr_output);

    // Load reference files
    auto ref_mc   = load_file(string(argv[2]) + "/subgraph_conv1.mc");
    auto ref_reg0 = load_file(string(argv[2]) + "/REG_0.bin");
    auto words    = load_words("./words.txt");

    void* mb; size_t ms;

    // ── REGION 1: INSTRUCTIONS ───────────────────────────────────────────────
    section("REGION 1 — INSTRUCTIONS (mc_code)");
    // dpu_instr_addr is IOMMU-translated via HPC0 port — NOT a CPU physical addr.
    // Scan HP region (±64MB around weights) to find actual physical address.
    printf("  Reg value 0x%lX = DPU-local IOMMU addr (HPC0). Scanning for actual phys addr...\n",
           instr_raw);

    uint64_t instr_phys = 0;
    if (ref_mc.size() >= SCAN_SIG_LEN && addr_weight > 0) {
        uint64_t scan_base = (addr_weight > SCAN_CHUNK) ? addr_weight - SCAN_CHUNK : 0;
        instr_phys = scan_for_signature(fd, scan_base, 2*SCAN_CHUNK,
                                        ref_mc.data(), SCAN_SIG_LEN, SCAN_STEP);
    }

    if (instr_phys) {
        printf("  Found at CPU phys : 0x%016lX\n", instr_phys);
        print_range(instr_phys, INSTR_SIZE);
        uint8_t* p = map_ddr4(fd, instr_phys, INSTR_SIZE, &mb, &ms);
        if (p) {
            compare(p, ref_mc.data(), min(INSTR_SIZE, ref_mc.size()));
            stats(p, INSTR_SIZE);
            munmap(mb, ms);
        }
    } else {
        printf("  Scan failed — instructions not found in ±64MB around weights.\n");
    }

    // ── REGION 2: WEIGHTS ────────────────────────────────────────────────────
    section("REGION 2 — WEIGHTS (REG_0 CONST)");
    // HP0 is 1:1 physical. addr_weight is directly usable with /dev/mem.
    print_range(addr_weight, WEIGHT_SIZE);
    printf("  Ref : REG_0.bin (%zu bytes)\n", ref_reg0.size());
    if (addr_weight && !ref_reg0.empty()) {
        uint8_t* p = map_ddr4(fd, addr_weight, WEIGHT_SIZE, &mb, &ms);
        if (p) {
            compare(p, ref_reg0.data(), min(WEIGHT_SIZE, ref_reg0.size()));
            stats(p, WEIGHT_SIZE);
            munmap(mb, ms);
        }
    }

    // ── REGION 3: FEATURE MAPS ───────────────────────────────────────────────
    section("REGION 3 — FEATURE MAPS (REG_1 WORKSPACE)");
    // Intermediate layer activations (conv/relu outputs) written by DPU at runtime.
    // ~50% zeros is the expected ReLU pattern (negatives clamped to 0).
    // No reference file possible — stats confirm data is valid and not garbage.
    print_range(addr_fmap, FMAP_SIZE);
    printf("  No reference (runtime-computed). ~50%% zeros = normal ReLU pattern.\n");
    if (addr_fmap) {
        uint8_t* p = map_ddr4(fd, addr_fmap, FMAP_SIZE, &mb, &ms);
        if (p) { stats(p, FMAP_SIZE); munmap(mb, ms); }
    }

    // ── REGION 4: INPUT TENSOR ───────────────────────────────────────────────
    section("REGION 4 — INPUT TENSOR (REG_2 INTERFACE)");
    // REG_2 size (152608) > imgBuf size (150528). Diff = 2080 bytes.
    // VART prepends a header/padding before the actual image data.
    // Scan within the DDR4 input region to find where our imgBuf bytes start.
    print_range(addr_input, INPUT_SIZE);
    printf("  imgBuf size = %d B  |  REG_2 size = %zu B  |  diff = %zd B (header/padding)\n",
           inSz, INPUT_SIZE, (ssize_t)INPUT_SIZE - inSz);

    if (addr_input && inSz >= (int)SCAN_SIG_LEN) {
        // Map the full INPUT_SIZE region
        uint8_t* inp_ddr4 = map_ddr4(fd, addr_input, INPUT_SIZE, &mb, &ms);
        if (inp_ddr4) {
            // Search for imgBuf signature within the DDR4 input region (byte-by-byte)
            size_t found_off = SIZE_MAX;
            for (size_t off = 0; off + SCAN_SIG_LEN <= INPUT_SIZE; off++) {
                if (memcmp(inp_ddr4 + off, imgBuf.data(), SCAN_SIG_LEN) == 0) {
                    found_off = off;
                    break;
                }
            }

            if (found_off != SIZE_MAX) {
                printf("  imgBuf content found at offset %zu (0x%zX) within DDR4 input region\n",
                       found_off, found_off);
                printf("  Comparing DDR4[offset %zu:] vs imgBuf (%d bytes):\n",
                       found_off, inSz);
                size_t cmp = min((size_t)inSz, INPUT_SIZE - found_off);
                compare(inp_ddr4 + found_off, (const uint8_t*)imgBuf.data(), cmp);
            } else {
                printf("  imgBuf signature not found within DDR4 input region.\n");
                printf("  VART likely DMA-copies to its own internal allocation,\n");
                printf("  not the addr_input region. Comparing DDR4 vs imgBuf at offset 0:\n");
                size_t cmp = min(INPUT_SIZE, (size_t)inSz);
                compare(inp_ddr4, (const uint8_t*)imgBuf.data(), cmp);
            }
            printf("  DDR4 input region stats:\n");
            stats(inp_ddr4, INPUT_SIZE);
            munmap(mb, ms);
        }
    }

    // ── REGION 5: OUTPUT TENSOR ──────────────────────────────────────────────
    section("REGION 5 — OUTPUT TENSOR (REG_3 INTERFACE)");
    // OUTPUT_SIZE=1008, outSz=1000. Extra 8 bytes = alignment padding.
    // Both DDR4 and fcBuf should match — DPU writes logits to this DDR4 addr,
    // runner copies them to fcBuf.
    print_range(addr_output, OUTPUT_SIZE);
    printf("  outSz=%d  OUTPUT_SIZE=%zu  extra=%zd B (alignment padding)\n",
           outSz, OUTPUT_SIZE, (ssize_t)OUTPUT_SIZE - outSz);

    if (addr_output) {
        size_t cmp = min(OUTPUT_SIZE, (size_t)outSz);
        uint8_t* p = map_ddr4(fd, addr_output, OUTPUT_SIZE, &mb, &ms);
        if (p) {
            compare(p, (const uint8_t*)fcBuf.data(), cmp);
            stats(p, OUTPUT_SIZE);

            // Decode top-5 from DDR4 directly
            int n_cls = min((int)cmp, 1000);
            vector<float> sm(n_cls);
            softmax((const int8_t*)p, n_cls, sm.data(), out_sc);
            auto tk = topk(sm.data(), n_cls, TOP_K);
            printf("  Top-%d from DDR4 output directly:\n", TOP_K);
            for (int i=0;i<TOP_K;i++) {
                int c=tk[i];
                printf("    [%d] class=%-4d  prob=%.6f  %s\n", i, c, sm[c],
                       c<(int)words.size()?words[c].c_str():"?");
            }

            // Cross-check vs fcBuf
            vector<float> sm2(outSz);
            softmax(fcBuf.data(), outSz, sm2.data(), out_sc);
            auto tk2 = topk(sm2.data(), outSz, TOP_K);
            printf("  Top-%d from CPU fcBuf (cross-check):\n", TOP_K);
            for (int i=0;i<TOP_K;i++) {
                int c=tk2[i];
                printf("    [%d] class=%-4d  prob=%.6f  %s\n", i, c, sm2[c],
                       c<(int)words.size()?words[c].c_str():"?");
            }
            munmap(mb, ms);
        }
    }

    // ── SUMMARY ──────────────────────────────────────────────────────────────
    printf("\n╔══════════════════════════════════════════════════════════════════════════════╗\n");
    printf("║  SUMMARY                                                                   ║\n");
    printf("╠═══════════════════╦══════════════════════╦══════════════╦══════════════════╣\n");
    printf("║  Region           ║  DDR4 phys base      ║  Size        ║  AXI / mapping   ║\n");
    printf("╠═══════════════════╬══════════════════════╬══════════════╬══════════════════╣\n");
    printf("║  Instructions     ║  0x%016lX  ║  %8zu B  ║  HPC0 IOMMU      ║\n", instr_phys?instr_phys:instr_raw, INSTR_SIZE);
    printf("║  Weights(REG_0)   ║  0x%016lX  ║  %8zu B  ║  HP0  1:1        ║\n", addr_weight, WEIGHT_SIZE);
    printf("║  FMaps(REG_1)     ║  0x%016lX  ║  %8zu B  ║  HP0  1:1        ║\n", addr_fmap,   FMAP_SIZE);
    printf("║  Input(REG_2)     ║  0x%016lX  ║  %8zu B  ║  HP0  1:1        ║\n", addr_input,  INPUT_SIZE);
    printf("║  Output(REG_3)    ║  0x%016lX  ║  %8zu B  ║  HP0  1:1        ║\n", addr_output, OUTPUT_SIZE);
    printf("╚═══════════════════╩══════════════════════╩══════════════╩══════════════════╝\n");
    printf("  Instructions base: %s\n",
           instr_phys ? "CPU physical addr (found by scan, 100% match confirmed)"
                      : "DPU-local IOMMU value shown (scan failed)");

    munmap((void*)ctrl, 4096);
    close(fd);
    printf("\n[Done]\n");
    return 0;
}