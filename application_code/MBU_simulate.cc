/*
 * SBU_simulate.cc  –  Single Bit Upset (SBU) Simulation
 * =======================================================
 *
 * HOW FAULT INJECTION ACTUALLY WORKS
 * ====================================
 * VART allocates DDR4 memory at startup and writes the addresses
 * into DPU registers (as shown in the diagram):
 *
 *   weights_ptr  = malloc(25MB)  -> e.g. 0x1bf00000
 *   input_ptr    = malloc(147KB) -> e.g. 0x1ba80000
 *   output_ptr   = malloc(4KB)   -> e.g. 0x05801000
 *
 *   write_register(0x80000060, weights_ptr)   // DPU weight address
 *   write_register(0x80000070, input_ptr)     // DPU input address
 *   write_register(0x80000078, output_ptr)    // DPU output address
 *   write_register(0x80000010, instr_ptr)     // DPU instruction address
 *
 * These CPU virtual addresses are accessible directly via:
 *
 *   INSTRUCTIONS: subgraph->get_attr<std::vector<char>>("mc_code")
 *                 mc_code.data() = CPU virtual addr of DPU instruction DDR4
 *
 *   WEIGHTS:      subgraph->get_attr<map<string,vector<char>>>(
 *                     "reg_id_to_parameter_value")
 *                 Each map value = CPU virtual addr of one weight DDR4 region
 *
 *   FEATURE_MAPS: imageInputs buffer passed to CpuFlatTensorBuffer
 *                 = the exact DDR4 buffer VART registered with DPU
 *
 *   BUFFERS:      FCResult buffer passed to CpuFlatTensorBuffer
 *                 = the exact DDR4 output buffer DPU writes to
 *
 * We flip bits directly in these CPU virtual addresses.
 * The DPU reads from those same DDR4 addresses during inference,
 * so it sees the corrupted data — no file reload needed.
 *
 * For INSTRUCTIONS: a single bit flip in mc_code typically causes
 * a DPU hang or exception (caught by timeout + HW reset, Solution 1+2).
 * For WEIGHTS: bit flips in weight data cause wrong conv outputs (SDC).
 * After each run we RESTORE the original bytes so the next run is clean.
 *
 * PROTECTION SOLUTIONS (all except checksum):
 *   [1] Timeout Detection   – kills hanging DPU inference
 *   [2] System / HW Reset   – recovers frozen DPU
 *   [3] Subgraph Tracking   – monitors DPU execution progress
 *   [4] Output Sanity Check – NaN / Inf / all-zero detection
 *
 * BUILD:
 *   g++ -std=c++17 -O2 -o SBU_simulate src/SBU_simulate.cc \
 *       ../common/common.cpp \
 *       -I ./src -I ../common \
 *       -I /usr/include/opencv4 -I /usr/include/vitis_ai \
 *       $(pkg-config --cflags --libs opencv4) \
 *       -lvart-runner -lxir -lglog -lpthread
 *
 * RUN:
 *   ./SBU_simulate <model.xmodel> [N] [k] [target] [-v]
 *   target: instructions | weights | feature_maps | buffers | all
 */

#include <assert.h>
#include <dirent.h>
#include <fcntl.h>
#include <signal.h>
#include <setjmp.h>
#include <stdio.h>
#include <stdlib.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <sys/wait.h>
#include <time.h>
#include <unistd.h>

#include <algorithm>
#include <atomic>
#include <chrono>
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
#include <string>
#include <thread>
#include <vector>

#include "common.h"
#include <opencv2/opencv.hpp>

using namespace std;
using namespace cv;
using namespace std::chrono;

// ─────────────────────────────────────────────────────────────────────────────
// SIGNAL HANDLER — catches SIGSEGV/SIGABRT from DPU driver crashes
// ─────────────────────────────────────────────────────────────────────────────
static sigjmp_buf  g_crash_jmp;
static volatile sig_atomic_t g_in_protected = 0;

static void crash_signal_handler(int sig) {
    if (g_in_protected) {
        siglongjmp(g_crash_jmp, sig);
    }
    ::signal(sig, SIG_DFL);
    ::raise(sig);
}

static void install_crash_handlers() {
    struct ::sigaction sa;
    memset(&sa, 0, sizeof(sa));
    sa.sa_handler = crash_signal_handler;
    sigemptyset(&sa.sa_mask);
    sa.sa_flags = SA_RESETHAND;
    ::sigaction(SIGSEGV, &sa, nullptr);
    ::sigaction(SIGABRT, &sa, nullptr);
    ::sigaction(SIGBUS,  &sa, nullptr);
}

static void reinstall_crash_handlers() {
    install_crash_handlers();
}

// ─────────────────────────────────────────────────────────────────────────────
// CONSTANTS
// ─────────────────────────────────────────────────────────────────────────────
#define INFERENCE_TIMEOUT_MS   10000
#define MAX_RECOVERY_ATTEMPTS  3
#define TOP_K                  5

// For INSTRUCTIONS fault injection: corrupt xmodel file → reload → create_runner()
// copies corrupted instructions into DPU on-chip SRAM.
// fault_injection_hybrid.cc used skip=100 bytes which crashes with many flips
// because protobuf string/length fields are in the first few KB.
// 64KB skip gives a safe margin — random flips land in weight/instruction data,
// not in protobuf structural bytes.
// If deserialization still fails (rare, very unlucky flip) → counted as DUE.
#define XMODEL_SAFE_SKIP       65536   // 64 KB

static const string baseImagePath       = "../images/";
static const string wordsPath           = "./";
static const string CORRUPTED_MODEL_PATH= "/tmp/sbu_corrupted.xmodel";

// ─────────────────────────────────────────────────────────────────────────────
// FAULT TARGET
// ─────────────────────────────────────────────────────────────────────────────
enum class FaultTarget {
    INSTRUCTIONS,  // DPU mc_code DDR4 region (instruction SRAM)
    WEIGHTS,       // DPU parameter DDR4 regions (weight SRAM)
    FEATURE_MAPS,  // Input tensor DDR4 buffer (activation map)
    BUFFERS,       // Output tensor DDR4 buffer (result accumulator)
    ALL            // random per run
};

static string targetName(FaultTarget t) {
    switch(t) {
        case FaultTarget::INSTRUCTIONS: return "INSTRUCTIONS";
        case FaultTarget::WEIGHTS:      return "WEIGHTS";
        case FaultTarget::FEATURE_MAPS: return "FEATURE_MAPS";
        case FaultTarget::BUFFERS:      return "BUFFERS";
        case FaultTarget::ALL:          return "ALL(random)";
    }
    return "UNKNOWN";
}

// ─────────────────────────────────────────────────────────────────────────────
// CONFIG
// ─────────────────────────────────────────────────────────────────────────────
struct SimConfig {
    string      model_path;
    int         N          = 100;
    int         k          = 1;
    FaultTarget target     = FaultTarget::FEATURE_MAPS;
    bool        verbose    = false;
    string      output_csv = "sbu_results.csv";
    string      log_file   = "sbu_sim.log";
};

// ─────────────────────────────────────────────────────────────────────────────
// PER-RUN RESULT
// ─────────────────────────────────────────────────────────────────────────────
struct RunResult {
    int         run_id             = 0;
    FaultTarget target_used        = FaultTarget::FEATURE_MAPS;
    int         bits_flipped       = 0;
    bool        timeout            = false;
    bool        crash              = false;
    bool        output_anomaly     = false;
    bool        top1_match         = false;
    bool        top5_match         = false;
    int         baseline_top1      = -1;
    int         faulty_top1        = -1;
    float       baseline_top1_prob = 0.f;
    float       faulty_top1_prob   = 0.f;
    float       prob_drop          = 0.f;
    bool        recovered          = false;
    uint64_t    fault_addr         = 0;   // DDR4 virtual address of first flip
    size_t      fault_byte_offset  = 0;   // offset within the region
    int         fault_bit          = 0;
    string      fault_region;
};

// ─────────────────────────────────────────────────────────────────────────────
// GLOBAL
// ─────────────────────────────────────────────────────────────────────────────
GraphInfo shapes;

struct GoldenRef {
    int          top1_class          = -1;
    int          top5_classes[TOP_K] = {};
    float        top1_prob           = 0.f;
    vector<float>softmax_vec;
    bool         valid               = false;
} g_golden;

struct SubgraphTracker {
    atomic<int>  completed{0};
    atomic<bool> timed_out{false};
    int          expected = 0;
    void reset(int n)    { completed=0; timed_out=false; expected=n; }
    void mark_complete() { completed++; }
} g_tracker;

// ─────────────────────────────────────────────────────────────────────────────
// LOGGING
// ─────────────────────────────────────────────────────────────────────────────
static FILE* g_logfp = nullptr;
static void sim_log(const char* fmt, ...) {
    va_list a1,a2;
    va_start(a1,fmt); vprintf(fmt,a1);       va_end(a1);
    if(g_logfp){ va_start(a2,fmt); vfprintf(g_logfp,fmt,a2); va_end(a2); fflush(g_logfp); }
}

// ─────────────────────────────────────────────────────────────────────────────
// SOLUTION 1 – TIMEOUT
// ─────────────────────────────────────────────────────────────────────────────
static int run_with_timeout(vart::Runner* runner,
                            vector<vart::TensorBuffer*>& ip,
                            vector<vart::TensorBuffer*>& op,
                            int timeout_ms) {
    atomic<int>  result{-1};
    atomic<bool> done{false};
    thread t([&](){
        try {
            auto job=runner->execute_async(ip,op);
            result=(runner->wait(job.first,timeout_ms)==0)?0:2;
        } catch(...){ result=2; }
        done=true;
    });
    auto t0=steady_clock::now();
    while(!done){
        this_thread::sleep_for(milliseconds(50));
        if(duration_cast<milliseconds>(steady_clock::now()-t0).count()>=timeout_ms){
            sim_log("[Timeout][1] DPU exceeded %d ms\n",timeout_ms);
            result=1; done=true; break;
        }
    }
    t.detach();
    return result.load();
}

// ─────────────────────────────────────────────────────────────────────────────
// SOLUTION 2 – HW/SW RESET
// ─────────────────────────────────────────────────────────────────────────────
static int hardware_reset_dpu() {
    sim_log("[Reset][2-HW] Attempting hardware DPU reset...\n");
    int fd=open("/dev/mem",O_RDWR|O_SYNC);
    if(fd>=0){
        void* b=mmap(NULL,4096,PROT_READ|PROT_WRITE,MAP_SHARED,fd,0x80000000);
        if(b!=MAP_FAILED){
            volatile uint32_t* c=(volatile uint32_t*)b;
            c[0]=0x00;usleep(10000);c[1]=0xFF;usleep(10000);
            c[0]=0x01;usleep(100000);c[0]=0x04;usleep(10000);
            munmap(b,4096);close(fd);
            sim_log("[Reset][2-HW] Register reset done\n"); return 0;
        }
        close(fd);
    }
    FILE* fp=fopen("/sys/class/dpu/dpu0/reset","w");
    if(fp){fprintf(fp,"1\n");fclose(fp);sleep(1);
           sim_log("[Reset][2-HW] Sysfs reset done\n"); return 0;}
    if(system("rmmod xrt_core 2>/dev/null;sleep 1;modprobe xrt_core 2>/dev/null")==0){
        sleep(2); sim_log("[Reset][2-HW] Module reload done\n"); return 0;}
    sim_log("[Reset][2-HW] All HW reset methods unavailable\n");
    return -1;
}

static unique_ptr<vart::Runner> recreate_runner(const xir::Subgraph* sg){
    sim_log("[Reset][2-SW] Recreating runner...\n");
    try{
        auto r=vart::Runner::create_runner(sg,"run");
        sim_log("[Reset][2-SW] OK\n"); return r;
    }catch(const exception& e){sim_log("[Reset][2-SW] FAIL: %s\n",e.what());}
     catch(...){sim_log("[Reset][2-SW] FAIL: unknown\n");}
    return nullptr;
}

// ─────────────────────────────────────────────────────────────────────────────
// SOLUTION 4 – OUTPUT SANITY
// ─────────────────────────────────────────────────────────────────────────────
static bool output_tensor_sane(const int8_t* d, int sz){
    if(sz<=0) return false;
    int zeros=0,mn=127,mx=-128;
    for(int i=0;i<sz;i++){
        int v=(int)d[i]; if(v==0)zeros++;
        if(v<mn)mn=v; if(v>mx)mx=v;
    }
    if(zeros==sz){sim_log("[Sanity][4] All-zero output\n");return false;}
    if(mn==mx)   {sim_log("[Sanity][4] Flat output (all=%d)\n",mn);return false;}
    return true;
}
static bool softmax_anomalous(const float* s, int sz){
    float sum=0.f;
    for(int i=0;i<sz;i++){if(!isfinite(s[i]))return true;sum+=s[i];}
    return sum<1e-3f;
}

// ─────────────────────────────────────────────────────────────────────────────
// PREPROCESSING / POSTPROCESSING
// ─────────────────────────────────────────────────────────────────────────────
static void preprocess_image(const Mat& src, int8_t* dst,
                              int inH, int inW, float scale){
    static const float mean[3]={104.f,107.f,123.f};
    Mat rsz; resize(src,rsz,Size(inW,inH),0,0,INTER_LINEAR);
    for(int h=0;h<inH;h++) for(int w=0;w<inW;w++){
        Vec3b px=rsz.at<Vec3b>(h,w);
        for(int c=0;c<3;c++){
            float v=((float)px[c]-mean[c])*scale;
            dst[h*inW*3+w*3+c]=(int8_t)max(-128.f,min(127.f,v));
        }
    }
}

static void CPUCalcSoftmax(const int8_t* d, int sz, float* out, float scale){
    double sum=0.0;
    for(int i=0;i<sz;i++){out[i]=expf((float)d[i]*scale);sum+=out[i];}
    for(int i=0;i<sz;i++) out[i]/=(float)sum;
}

static vector<int> topk(const float* p, int sz, int k){
    vector<int> idx(sz); iota(idx.begin(),idx.end(),0);
    partial_sort(idx.begin(),idx.begin()+k,idx.end(),
                 [&](int a,int b){return p[a]>p[b];});
    idx.resize(k); return idx;
}

// ─────────────────────────────────────────────────────────────────────────────
// FILE HELPERS
// ─────────────────────────────────────────────────────────────────────────────
static void ListImages(const string& path, vector<string>& images){
    images.clear();
    struct stat s; lstat(path.c_str(),&s);
    if(!S_ISDIR(s.st_mode)){fprintf(stderr,"[Error] %s not a dir\n",path.c_str());exit(1);}
    DIR* dir=opendir(path.c_str());
    if(!dir){fprintf(stderr,"[Error] can't open %s\n",path.c_str());exit(1);}
    struct dirent* e;
    while((e=readdir(dir))){
        if(e->d_type==DT_REG||e->d_type==DT_UNKNOWN){
            string n=e->d_name; if(n.size()<4)continue;
            string ext=n.substr(n.find_last_of('.')+1);
            transform(ext.begin(),ext.end(),ext.begin(),::tolower);
            if(ext=="jpg"||ext=="jpeg"||ext=="png")images.push_back(n);
        }
    }
    closedir(dir); sort(images.begin(),images.end());
}
static void LoadWords(const string& path, vector<string>& kinds){
    kinds.clear(); ifstream f(path);
    if(!f){fprintf(stderr,"[Error] can't open %s\n",path.c_str());exit(1);}
    string line; while(getline(f,line))kinds.push_back(line);
}

// ─────────────────────────────────────────────────────────────────────────────
// SBU FLIP PRIMITIVE  (operates on a raw byte pointer)
// ─────────────────────────────────────────────────────────────────────────────
struct FlipInfo { size_t offset; int bit; uint8_t before; uint8_t after; };

static vector<FlipInfo> inject_sbu(uint8_t* base, size_t region_sz,
                                    int k, mt19937& rng, bool verbose,
                                    const char* tag){
    vector<FlipInfo> flips;
    if(!base||region_sz==0||k<=0) return flips;
    uniform_int_distribution<size_t> bdist(0,region_sz-1);
    uniform_int_distribution<int>    bitdist(0,7);
    set<size_t> used;
    int tries=0;
    while((int)flips.size()<k && tries<k*20){
        size_t off=bdist(rng);
        if(used.count(off)){tries++;continue;}
        used.insert(off);
        int     bit=bitdist(rng);
        uint8_t orig=base[off];
        base[off]^=(uint8_t)(1u<<bit);
        flips.push_back({off,bit,orig,base[off]});
        if(verbose)
            sim_log("  [SBU][%s] offset=%7zu bit%d 0x%02X->0x%02X (diff=%+d)\n",
                    tag,off,bit,orig,base[off],(int)base[off]-(int)orig);
        tries++;
    }
    return flips;
}

static void restore_flips(uint8_t* base, const vector<FlipInfo>& flips){
    for(auto& f:flips) base[f.offset]=f.before;
}

// ─────────────────────────────────────────────────────────────────────────────
// DDR4 REGION ACCESS VIA XIR SUBGRAPH ATTRIBUTES
// ─────────────────────────────────────────────────────────────────────────────
/*
 * XIR stores the DPU memory regions as subgraph attributes.
 * We enumerate ALL attributes at startup so we can see exactly what the
 * real hardware exposes, then pick the right ones for fault injection.
 *
 * Known attribute names across Vitis AI versions:
 *   "mc_code"                    vector<char>             – instruction stream
 *   "reg_id_to_parameter_value"  map<string,vector<char>> – weight regions
 *   "reg_id_to_context_type"     map<string,string>       – REG type: PARAM/CODE/IO
 *   "reg_id_to_code"             map<string,vector<char>> – alt instruction attr
 */

struct RegionHandle {
    uint8_t* ptr  = nullptr;
    size_t   size = 0;
    string   name;
};

// ── Dump every subgraph attribute with its size ───────────────────────────────
// Call this once at startup. Output tells us exactly what is available
// on this specific VART/XIR version and model.
static void dump_subgraph_attrs(const xir::Subgraph* sg) {
    sim_log("\n[AttrDump] Subgraph: %s\n", sg->get_name().c_str());
    sim_log("[AttrDump] Probing known XIR attributes:\n");

    // Probe all known attribute names — avoids get_attr_names()/get_attrs()
    // version differences across Vitis AI releases.
    static const char* known_vec_char[] = {
        "mc_code", "elf_code", "dpu_fingerprint", nullptr
    };
    static const char* known_map_vec[] = {
        "reg_id_to_parameter_value", "reg_id_to_code",
        "reg_id_to_initial_value", nullptr
    };
    static const char* known_map_str[] = {
        "reg_id_to_context_type", "reg_id_to_type", nullptr
    };
    static const char* known_int[] = {
        "workload", "depth", nullptr
    };
    static const char* known_str[] = {
        "device", "dpu_arch", "dpu_type", nullptr
    };

    for (int i = 0; known_vec_char[i]; i++) {
        try {
            auto& v = sg->get_attr<vector<char>>(known_vec_char[i]);
            sim_log("[AttrDump]   %-40s  vector<char>  size=%zu bytes\n",
                    known_vec_char[i], v.size());
        } catch(...) {
            sim_log("[AttrDump]   %-40s  (not present)\n", known_vec_char[i]);
        }
    }
    for (int i = 0; known_map_vec[i]; i++) {
        try {
            auto& m = sg->get_attr<map<string,vector<char>>>(known_map_vec[i]);
            sim_log("[AttrDump]   %-40s  map  entries=%zu\n",
                    known_map_vec[i], m.size());
            for (auto& [k,v] : m)
                sim_log("[AttrDump]     key=%-16s  size=%zu bytes\n", k.c_str(), v.size());
        } catch(...) {
            sim_log("[AttrDump]   %-40s  (not present)\n", known_map_vec[i]);
        }
    }
    for (int i = 0; known_map_str[i]; i++) {
        try {
            auto& m = sg->get_attr<map<string,string>>(known_map_str[i]);
            sim_log("[AttrDump]   %-40s  map  entries=%zu\n",
                    known_map_str[i], m.size());
            for (auto& [k,v] : m)
                sim_log("[AttrDump]     key=%-16s  val=%s\n", k.c_str(), v.c_str());
        } catch(...) {
            sim_log("[AttrDump]   %-40s  (not present)\n", known_map_str[i]);
        }
    }
    for (int i = 0; known_int[i]; i++) {
        try {
            auto v = sg->get_attr<int32_t>(known_int[i]);
            sim_log("[AttrDump]   %-40s  int32=%d\n", known_int[i], v);
        } catch(...) {}
    }
    for (int i = 0; known_str[i]; i++) {
        try {
            auto v = sg->get_attr<string>(known_str[i]);
            sim_log("[AttrDump]   %-40s  string=\"%s\"\n", known_str[i], v.c_str());
        } catch(...) {}
    }
    sim_log("\n");
}

// ── Get instruction region ────────────────────────────────────────────────────
// Tries multiple known attr names in priority order.
// Uses .size() of the vector<char> as the EXACT instruction byte count.
static RegionHandle get_instruction_region(const xir::Subgraph* sg) {
    RegionHandle h;

    // Priority 1: "mc_code"
    try {
        auto& mc = sg->get_attr<vector<char>>("mc_code");
        if (!mc.empty()) {
            h.ptr  = reinterpret_cast<uint8_t*>(const_cast<char*>(mc.data()));
            h.size = mc.size();
            h.name = "INSTRUCTIONS(mc_code)";
            sim_log("[DDR4] Instructions via 'mc_code': size=%zu bytes\n", h.size);
            return h;
        }
    } catch(...) {}

    // Priority 2: "reg_id_to_context_type" — find CODE-type registers
    try {
        auto& ctx_map = sg->get_attr<map<string,string>>("reg_id_to_context_type");
        auto& val_map = sg->get_attr<map<string,vector<char>>>("reg_id_to_parameter_value");
        for (auto& [reg_id, ctx_type] : ctx_map) {
            if (ctx_type == "CODE" || ctx_type == "code" ||
                ctx_type == "INSTRUCTION" || ctx_type == "MC_CODE") {
                if (val_map.count(reg_id) && !val_map.at(reg_id).empty()) {
                    auto& d = val_map.at(reg_id);
                    h.ptr  = reinterpret_cast<uint8_t*>(const_cast<char*>(d.data()));
                    h.size = d.size();
                    h.name = "INSTRUCTIONS(ctx_type=" + reg_id + ")";
                    sim_log("[DDR4] Instructions via context_type CODE reg=%s: "
                            "vaddr=0x%016lX  size=%zu bytes\n",
                            reg_id.c_str(), (uint64_t)h.ptr, h.size);
                    return h;
                }
            }
        }
    } catch(...) {}

    // Priority 3: "reg_id_to_code"
    try {
        auto& code_map = sg->get_attr<map<string,vector<char>>>("reg_id_to_code");
        for (auto& [reg_id, data] : code_map) {
            if (!data.empty()) {
                h.ptr  = reinterpret_cast<uint8_t*>(const_cast<char*>(data.data()));
                h.size = data.size();
                h.name = "INSTRUCTIONS(reg_id_to_code[" + reg_id + "])";
                sim_log("[DDR4] Instructions via 'reg_id_to_code' reg=%s: "
                        "vaddr=0x%016lX  size=%zu bytes\n",
                        reg_id.c_str(), (uint64_t)h.ptr, h.size);
                return h;
            }
        }
    } catch(...) {}

    sim_log("[DDR4] WARNING: No instruction region found.\n");
    sim_log("[DDR4] Check AttrDump above for the correct attr name.\n");
    return h;
}

// ── Force DPU to re-fetch instructions from DDR4 (ZCU104 / DPUCZDX8G) ────────
//
// On ZCU104 + DPUCZDX8G (Vitis-AI 3.0) the DPU copies mc_code into its
// on-chip instruction memory at create_runner() time.  After that the PS
// DDR4 mc_code region is NOT re-read during inference — so flipping bits
// there has no effect.
//
// Fix (from Grok analysis + DPU register map):
//   Write the mc_code DDR4 virtual address into DPU registers 4 and 5
//   (high-32 and low-32 of the 64-bit instruction base address).
//   This forces the DPU to reload instructions from DDR4 on the next
//   execute_async(), bypassing the on-chip cache.
//
//   DPU base register addresses for ZCU104 + DPUCZDX8G (Vitis-AI 2.x/3.0):
//     Primary:   0x8F000000  (DPUCZDX8G default in device tree)
//     Alternate: 0x80000000  (used in some board configurations)
//
//   Within the DPU register bank, the instruction address registers are:
//     Offset 0x208 = INSTR_ADDR_L  (low  32-bit of instruction base)
//     Offset 0x20C = INSTR_ADDR_H  (high 32-bit of instruction base)
//   (From Xilinx PG338 DPU TRM, Table: DPU Control Registers)
//
// Per-run independence: flip → reload corrupted → inference → restore → reload clean
// Each run independently picks new random bit positions in mc_code DDR4.
//
// Returns true if register write succeeded (false = platform limitation,
// instruction faults will have no effect on this run).
static bool force_dpu_reload_instructions(uint64_t instr_vaddr) {
    // Try all known DPU base addresses for ZCU104 / Versal / Zynq platforms
    static const uint32_t dpu_bases[] = {
        0x8F000000,   // DPUCZDX8G default (ZCU104, ZCU102, Ultra96)
        0x80000000,   // alternate / older board configs
        0x8FF00000,   // some custom overlays
    };
    // Instruction address register offsets (PG338 Table 2-6)
    static const uint32_t INSTR_ADDR_L = 0x208;  // low  32-bit
    static const uint32_t INSTR_ADDR_H = 0x20C;  // high 32-bit

    int fd = open("/dev/mem", O_RDWR | O_SYNC);
    if (fd < 0) {
        sim_log("[InstrReload] ERROR: Cannot open /dev/mem (%s)\n"
                "[InstrReload] Instruction fault injection NOT active on this run.\n",
                strerror(errno));
        return false;
    }

    bool wrote = false;
    for (uint32_t base_addr : dpu_bases) {
        void* b = mmap(NULL, 4096, PROT_READ|PROT_WRITE,
                       MAP_SHARED, fd, (off_t)base_addr);
        if (b == MAP_FAILED) continue;

        volatile uint8_t* regs = (volatile uint8_t*)b;

        // Write low-32 first, then high-32 (matches DPU register update sequence)
        *((volatile uint32_t*)(regs + INSTR_ADDR_L)) =
            (uint32_t)(instr_vaddr & 0xFFFFFFFF);
        *((volatile uint32_t*)(regs + INSTR_ADDR_H)) =
            (uint32_t)((instr_vaddr >> 32) & 0xFFFFFFFF);
        __sync_synchronize();

        sim_log("[InstrReload] Wrote instr ptr 0x%016lX to DPU base 0x%08X "
                "(offset L=0x%03X H=0x%03X)\n",
                instr_vaddr, base_addr, INSTR_ADDR_L, INSTR_ADDR_H);

        munmap(b, 4096);
        wrote = true;
        break;  // stop at first successful mmap
    }
    close(fd);

    if (!wrote)
        sim_log("[InstrReload] ERROR: All DPU base addresses failed to mmap.\n"
                "[InstrReload] Instruction fault injection NOT active on this run.\n");
    return wrote;
}

// ── Get weight region names and sizes only (no raw pointers — get_attr by value) ─
// Returns map of reg_id → size for logging. Actual data accessed via get_attr copy.
static void log_weight_regions(const xir::Subgraph* sg) {
    try {
        auto val_map = sg->get_attr<map<string,vector<char>>>(
                           "reg_id_to_parameter_value");
        for (auto& [reg_id, data] : val_map) {
            if (data.empty()) continue;
            sim_log("[DDR4] Weight region %-12s  size=%zu bytes\n",
                    reg_id.c_str(), data.size());
        }
    } catch(const exception& e) {
        sim_log("[DDR4] log_weight_regions: %s\n", e.what());
    }
}

// ── Find byte offset of needle inside haystack ────────────────────────────────
// Used to locate mc_code bytes inside the xmodel binary for surgical patching.
// Returns offset of first match, or string::npos if not found.
static size_t find_bytes_offset(const vector<uint8_t>& haystack,
                                 const vector<char>& needle,
                                 size_t search_start = 0) {
    if(needle.empty() || needle.size() > haystack.size()) return string::npos;
    // Search using first 32 bytes as signature to keep it fast
    size_t sig_len = min((size_t)32, needle.size());
    const uint8_t* h = haystack.data() + search_start;
    size_t h_len = haystack.size() - search_start;
    for(size_t i = 0; i + sig_len <= h_len; i++){
        if(memcmp(h + i, needle.data(), sig_len) == 0){
            // Verify full match (first 256 bytes)
            size_t verify_len = min((size_t)256, needle.size());
            if(memcmp(h + i, needle.data(), verify_len) == 0)
                return search_start + i;
        }
    }
    return string::npos;
}

// ─────────────────────────────────────────────────────────────────────────────
// Cached file offsets for INSTRUCTIONS and WEIGHTS (computed once at startup)
// Avoids 25MB get_attr copy + find_bytes_offset search every single run.
// ─────────────────────────────────────────────────────────────────────────────
static size_t g_mc_code_offset = string::npos;
static size_t g_mc_code_size   = 0;
static size_t g_weight_offset  = string::npos;
static size_t g_weight_size    = 0;

static void cache_region_offsets(const xir::Subgraph* sg,
                                  const vector<uint8_t>& clean_model) {
    auto* sgm = const_cast<xir::Subgraph*>(sg);

    // mc_code
    try {
        auto mc = sgm->get_attr<vector<char>>("mc_code");
        if(!mc.empty()){
            g_mc_code_size   = mc.size();
            g_mc_code_offset = find_bytes_offset(clean_model, mc);
            if(g_mc_code_offset != string::npos)
                sim_log("[Cache] mc_code: file_offset=%zu  size=%zu bytes\n",
                        g_mc_code_offset, g_mc_code_size);
            else
                sim_log("[Cache] mc_code: NOT FOUND in binary\n");
        }
    } catch(...) { sim_log("[Cache] mc_code attr not available\n"); }

    // weights (first non-empty entry of reg_id_to_parameter_value)
    try {
        auto wmap = sgm->get_attr<map<string,vector<char>>>(
                        "reg_id_to_parameter_value");
        for(auto& [rid, d] : wmap){
            if(d.empty()) continue;
            g_weight_size   = d.size();
            g_weight_offset = find_bytes_offset(clean_model, d);
            if(g_weight_offset != string::npos)
                sim_log("[Cache] weights[%s]: file_offset=%zu  size=%zu bytes\n",
                        rid.c_str(), g_weight_offset, g_weight_size);
            else
                sim_log("[Cache] weights[%s]: NOT FOUND in binary\n", rid.c_str());
            break;
        }
    } catch(...) { sim_log("[Cache] reg_id_to_parameter_value attr not available\n"); }
}

// ─────────────────────────────────────────────────────────────────────────────
// ─────────────────────────────────────────────────────────────────────────────
struct InferenceResult {
    bool     ok            = false;
    bool     timed_out     = false;
    bool     exception     = false;
    bool     output_bad    = false;
    int      top1          = -1;
    float    top1_prob     = 0.f;
    int      top5[TOP_K]   = {};
};

static InferenceResult run_inference(
        vart::Runner* runner,
        int8_t* imgBuf, int inSz, int inH, int inW,
        int8_t* fcBuf,  int outSz,
        float out_scale,
        const xir::Tensor* inTensor,
        const xir::Tensor* outTensor)
{
    InferenceResult R;

    auto idims = inTensor->get_shape();  idims[0]=1;
    auto odims = outTensor->get_shape(); odims[0]=1;

    vector<unique_ptr<vart::TensorBuffer>> ibuf, obuf;
    vector<shared_ptr<xir::Tensor>>        bt;

    bt.push_back(shared_ptr<xir::Tensor>(
        xir::Tensor::create(inTensor->get_name(), idims,
                            xir::DataType{xir::DataType::XINT,8u})));
    ibuf.push_back(make_unique<CpuFlatTensorBuffer>(imgBuf, bt.back().get()));

    bt.push_back(shared_ptr<xir::Tensor>(
        xir::Tensor::create(outTensor->get_name(), odims,
                            xir::DataType{xir::DataType::XINT,8u})));
    obuf.push_back(make_unique<CpuFlatTensorBuffer>(fcBuf, bt.back().get()));

    vector<vart::TensorBuffer*> ip={ibuf[0].get()}, op={obuf[0].get()};

    // Solution 3: track subgraph
    g_tracker.reset(1);

    // Solution 1: timeout
    int ret = run_with_timeout(runner, ip, op, INFERENCE_TIMEOUT_MS);

    if(ret==1){ R.timed_out=true;  return R; }
    if(ret==2){ R.exception=true;  return R; }

    g_tracker.mark_complete();  // Solution 3

    // Solution 4: sanity
    if(!output_tensor_sane(fcBuf,outSz)){ R.output_bad=true; }

    vector<float> sm(outSz);
    CPUCalcSoftmax(fcBuf,outSz,sm.data(),out_scale);
    if(softmax_anomalous(sm.data(),outSz)){ R.output_bad=true; }

    auto tk=topk(sm.data(),outSz,TOP_K);
    R.top1=tk[0]; R.top1_prob=sm[tk[0]];
    for(int i=0;i<TOP_K;i++) R.top5[i]=tk[i];
    R.ok=true;
    return R;
}

// ─────────────────────────────────────────────────────────────────────────────
// GOLDEN BASELINE
// ─────────────────────────────────────────────────────────────────────────────
static bool run_golden_baseline(vart::Runner* runner,
                                const vector<string>& images,
                                const vector<string>& kinds){
    sim_log("\n[Baseline] Running golden reference (no faults)...\n");

    auto outT=runner->get_output_tensors();
    auto inT =runner->get_input_tensors();
    float in_scale =get_input_scale(inT[0]);
    float out_scale=get_output_scale(outT[0]);
    int outSz=shapes.outTensorList[0].size;
    int inSz =shapes.inTensorList[0].size;
    int inH  =shapes.inTensorList[0].height;
    int inW  =shapes.inTensorList[0].width;

    vector<int8_t> imgBuf(inSz), fcBuf(outSz);

    Mat raw=imread(baseImagePath+images[0]);
    if(raw.empty()){sim_log("[Baseline] Cannot read %s\n",images[0].c_str());return false;}
    preprocess_image(raw,imgBuf.data(),inH,inW,in_scale);

    auto R=run_inference(runner,
                         imgBuf.data(),inSz,inH,inW,
                         fcBuf.data(),outSz,out_scale,
                         inT[0],outT[0]);

    if(!R.ok||R.timed_out||R.exception||R.output_bad){
        sim_log("[Baseline] Failed\n"); return false;}

    g_golden.top1_class=R.top1;
    g_golden.top1_prob =R.top1_prob;
    g_golden.valid     =true;
    for(int i=0;i<TOP_K;i++) g_golden.top5_classes[i]=R.top5[i];

    sim_log("[Baseline] Top-1: class=%-4d (%s)  prob=%.4f\n",
            R.top1,
            (R.top1<(int)kinds.size())?kinds[R.top1].c_str():"?",
            R.top1_prob);
    sim_log("[Baseline] Top-5:");
    for(int i=0;i<TOP_K;i++) sim_log(" %d",R.top5[i]);
    sim_log("\n");
    return true;
}

// ─────────────────────────────────────────────────────────────────────────────
// SUBPROCESS INFERENCE — isolates VART abort() from the main process
//
// When corrupted instructions/weights cause a DPU hang, VART's worker thread
// calls LOG(FATAL) → abort(). This cannot be caught by try/catch or signal
// handlers since it originates in a background thread.
//
// Solution: fork() a child. Child runs the corrupted inference and may die
// from VART's abort(). Parent detects child death, resets DPU IP via /dev/mem
// (zocl already cleaned up ERT when child exited), recreates main runner.
// ─────────────────────────────────────────────────────────────────────────────
static bool run_in_subprocess(
    const string& corrupted_path,
    const int8_t* imgBuf,
    int outSz, float out_scale,
    int run_id, bool verbose,
    vart::Runner*& runner,
    const xir::Subgraph* sg,
    RunResult& RES)
{
    int pfd[2];
    if(pipe(pfd)!=0){ sim_log("[Run %4d] pipe: %s\n",run_id,strerror(errno));
                      RES.crash=true; return false; }

    pid_t pid = fork();
    if(pid<0){ close(pfd[0]); close(pfd[1]);
               sim_log("[Run %4d] fork: %s\n",run_id,strerror(errno));
               RES.crash=true; return false; }

    if(pid==0){
        // ── CHILD ─────────────────────────────────────────────────────────
        close(pfd[0]);
        auto die=[&](){ write(pfd[1],"DUE\n",4); close(pfd[1]); _exit(1); };
        try {
            auto cg   = xir::Graph::deserialize(corrupted_path);
            auto csgv = get_dpu_subgraph(cg.get());
            if(csgv.empty()) die();
            auto cr = vart::Runner::create_runner(csgv[0],"run");

            auto coutT=cr->get_output_tensors(), cinT=cr->get_input_tensors();
            auto codims=coutT[0]->get_shape(); codims[0]=1;
            auto cidims=cinT[0]->get_shape();  cidims[0]=1;
            int coutSz=(int)coutT[0]->get_data_size();
            if(coutSz<=0||coutSz>100000) coutSz=outSz;

            vector<int8_t> cfcBuf(coutSz,0);
            vector<unique_ptr<vart::TensorBuffer>> cib,cob;
            vector<shared_ptr<xir::Tensor>> cbt;
            cbt.push_back(shared_ptr<xir::Tensor>(xir::Tensor::create(
                cinT[0]->get_name(),cidims,xir::DataType{xir::DataType::XINT,8u})));
            cib.push_back(make_unique<CpuFlatTensorBuffer>(
                const_cast<int8_t*>(imgBuf),cbt.back().get()));
            cbt.push_back(shared_ptr<xir::Tensor>(xir::Tensor::create(
                coutT[0]->get_name(),codims,xir::DataType{xir::DataType::XINT,8u})));
            cob.push_back(make_unique<CpuFlatTensorBuffer>(cfcBuf.data(),cbt.back().get()));
            vector<vart::TensorBuffer*> ip={cib[0].get()},op={cob[0].get()};

            // No software timeout — VART's XLNX_DPU_TIMEOUT will abort() this
            // child if the DPU hangs; the parent process is unaffected.
            auto fut=cr->execute_async(ip,op);
            cr->wait(fut.first,-1);

            vector<float> sm(outSz,0.f),tmp(coutSz);
            CPUCalcSoftmax(cfcBuf.data(),coutSz,tmp.data(),out_scale);
            int n=min(coutSz,outSz); for(int i=0;i<n;i++) sm[i]=tmp[i];
            auto tk=topk(sm.data(),outSz,TOP_K);
            char buf[256];
            int len=snprintf(buf,sizeof(buf),"OK %d %.8f %d %d %d %d %d\n",
                tk[0],sm[tk[0]],tk[0],tk[1],tk[2],tk[3],tk[4]);
            write(pfd[1],buf,len);
            close(pfd[1]);
            _exit(0);
        } catch(...){ die(); }
    }

    // ── PARENT ────────────────────────────────────────────────────────────
    close(pfd[1]);
    // Wait up to VART DPU timeout + 15s margin
    const int WAIT_MS = INFERENCE_TIMEOUT_MS + 15000;
    auto t0c = chrono::steady_clock::now();
    int wst=0; pid_t wr=0;
    while(true){
        wr=waitpid(pid,&wst,WNOHANG);
        if(wr==pid) break;
        if(chrono::duration_cast<chrono::milliseconds>(
               chrono::steady_clock::now()-t0c).count()>WAIT_MS){
            kill(pid,SIGKILL); waitpid(pid,&wst,0); break;
        }
        this_thread::sleep_for(chrono::milliseconds(100));
    }
    char rbuf[256]={}; read(pfd[0],rbuf,sizeof(rbuf)-1);
    close(pfd[0]);
    unlink(corrupted_path.c_str());

    bool ok = (WIFEXITED(wst)&&WEXITSTATUS(wst)==0);
    if(!ok){
        // Child died (DPU hung → VART abort, or deserialization crash).
        // zocl cleaned up ERT on child exit. Reset DPU IP via /dev/mem.
        sim_log("[Run %4d] Subprocess died — DUE. Resetting DPU IP...\n",run_id);
        hardware_reset_dpu();
        this_thread::sleep_for(chrono::milliseconds(2000));
        // Recreate main runner so parent's XRT context is fresh.
        auto nr=recreate_runner(sg);
        if(nr){ runner=nr.release();
                sim_log("[Run %4d] Main runner recreated\n",run_id); }
        RES.timeout=true; return true;
    }

    if(strncmp(rbuf,"OK ",3)!=0){ RES.crash=true; return true; }
    int ftop1,t0r,t1r,t2r,t3r,t4r; float fprob;
    if(sscanf(rbuf+3,"%d %f %d %d %d %d %d",
              &ftop1,&fprob,&t0r,&t1r,&t2r,&t3r,&t4r)<2){
        RES.crash=true; return true;
    }
    RES.faulty_top1=ftop1; RES.faulty_top1_prob=fprob;
    RES.prob_drop=g_golden.top1_prob-fprob;
    RES.top1_match=(ftop1==g_golden.top1_class);
    int ftk5[5]={t0r,t1r,t2r,t3r,t4r};
    for(int i=0;i<5;i++) if(ftk5[i]==g_golden.top1_class){RES.top5_match=true;break;}
    if(verbose)
        sim_log("[Run %4d] base=%-4d(%.4f) faulty=%-4d(%.4f) %s\n",
                run_id,g_golden.top1_class,g_golden.top1_prob,
                ftop1,fprob,RES.top1_match?"MATCH":"MISMATCH");
    return true;
}

// ─────────────────────────────────────────────────────────────────────────────
// SINGLE FAULTY RUN
// ─────────────────────────────────────────────────────────────────────────────
static bool perform_faulty_run(vart::Runner*& runner,
                               const xir::Subgraph* sg,
                               const vector<uint8_t>& clean_model,
                               const vector<string>& images,
                               const vector<string>& kinds,
                               const SimConfig& cfg,
                               int run_id,
                               mt19937& rng,
                               RunResult& RES){
    // Resolve target
    FaultTarget eff=cfg.target;
    if(eff==FaultTarget::ALL){
        static const FaultTarget pool[]={
            FaultTarget::INSTRUCTIONS,FaultTarget::WEIGHTS,
            FaultTarget::FEATURE_MAPS,FaultTarget::BUFFERS};
        eff=pool[rng()%4];
    }

    RES={};
    RES.run_id            =run_id;
    RES.target_used       =eff;
    RES.bits_flipped      =cfg.k;
    RES.baseline_top1     =g_golden.top1_class;
    RES.baseline_top1_prob=g_golden.top1_prob;
    RES.faulty_top1       =-1;
    RES.fault_region      =targetName(eff);

    if(cfg.verbose)
        sim_log("\n-- Run %4d | target=%-14s | k=%d --\n",
                run_id,targetName(eff).c_str(),cfg.k);

    auto outT =runner->get_output_tensors();
    auto inT  =runner->get_input_tensors();
    float in_scale =get_input_scale(inT[0]);
    float out_scale=get_output_scale(outT[0]);
    int outSz=shapes.outTensorList[0].size;
    int inSz =shapes.inTensorList[0].size;
    int inH  =shapes.inTensorList[0].height;
    int inW  =shapes.inTensorList[0].width;

    vector<int8_t> imgBuf(inSz,0), fcBuf(outSz,0);

    // Preprocess
    Mat raw=imread(baseImagePath+images[run_id%(int)images.size()]);
    if(raw.empty()){
        sim_log("[Run %4d] Cannot read image\n",run_id);
        RES.crash=true; return false;
    }
    preprocess_image(raw,imgBuf.data(),inH,inW,in_scale);

    // ── FAULT INJECTION ───────────────────────────────────────────────────────
    vector<FlipInfo> flips;
    uint8_t* flip_base = nullptr;
    size_t   flip_size = 0;
    string   flip_tag;

    if(eff==FaultTarget::INSTRUCTIONS || eff==FaultTarget::WEIGHTS){
        // ── Unified file-patch approach for both INSTRUCTIONS and WEIGHTS ────
        //
        // INSTRUCTIONS: mc_code bytes are 742492 bytes in the file.
        //   DPU on-chip SRAM is programmed at first create_runner and persists.
        //   We must hardware_reset_dpu() to clear on-chip SRAM, then load fresh.
        //
        // WEIGHTS: REG_0 is 25.7MB — too large for on-chip SRAM, always in DDR4.
        //   create_runner(new_subgraph) allocates DDR4 and DMAs weights from file.
        //   We must NOT call create_runner(original_sg) — that segfaults because
        //   a runner already exists on it. Use csg[0] from the new deserialized graph.
        //
        // Shared procedure:
        //   1. find attr bytes in clean_model binary
        //   2. flip k bits only within that byte range
        //   3. write patched file to /tmp/
        //   4. [INSTRUCTIONS only] hardware reset to clear on-chip SRAM
        //   5. deserialize + create_runner(csg[0])
        //   6. run inference, record result

        if(clean_model.empty()){
            sim_log("[Run %4d] clean_model not loaded\n",run_id);
            RES.crash=true; return false;
        }

        // Use cached file offsets (computed once at startup by cache_region_offsets)
        size_t attr_offset, attr_size;
        string attr_name;
        if(eff==FaultTarget::INSTRUCTIONS){
            attr_offset = g_mc_code_offset;
            attr_size   = g_mc_code_size;
            attr_name   = "mc_code";
        } else {
            attr_offset = g_weight_offset;
            attr_size   = g_weight_size;
            attr_name   = "WEIGHTS[REG_0]";
        }

        if(attr_offset == string::npos || attr_size == 0){
            sim_log("[Run %4d] %s region not cached — skipping\n",run_id,attr_name.c_str());
            RES.crash=true; return false;
        }

        // Make per-run copy, flip k bits only within attr payload range
        vector<uint8_t> patched = clean_model;
        if(cfg.verbose)
            sim_log("  [Patch] %s at file offset %zu  size=%zu bytes\n",
                    attr_name.c_str(), attr_offset, attr_size);
        vector<FlipInfo> pf = inject_sbu(
            patched.data() + attr_offset, attr_size,
            cfg.k, rng, cfg.verbose, attr_name.c_str());
        if(!pf.empty()){
            RES.fault_byte_offset = attr_offset + pf[0].offset;
            RES.fault_addr        = RES.fault_byte_offset;
            RES.fault_bit         = pf[0].bit;
        }

        // Write patched file
        {
            ofstream out(CORRUPTED_MODEL_PATH, ios::binary);
            if(!out){
                sim_log("[Run %4d] Cannot write %s\n",run_id,CORRUPTED_MODEL_PATH.c_str());
                RES.crash=true; return false;
            }
            out.write(reinterpret_cast<const char*>(patched.data()), patched.size());
        }

        // Run corrupted inference in a subprocess so VART's abort() on DPU hang
        // kills only the child, not the main simulator process.
        return run_in_subprocess(
            CORRUPTED_MODEL_PATH,
            imgBuf.data(), outSz, out_scale,
            run_id, cfg.verbose,
            runner, sg, RES);

    } else if(eff==FaultTarget::FEATURE_MAPS){
        // Flip bits in the input DDR4 buffer (imageInputs in main.cc terms)
        flip_base=reinterpret_cast<uint8_t*>(imgBuf.data());
        flip_size=(size_t)inSz;
        flip_tag ="feature_maps";
        flips=inject_sbu(flip_base,flip_size,cfg.k,rng,cfg.verbose,flip_tag.c_str());
        if(!flips.empty()){
            RES.fault_addr       =(uint64_t)flip_base+flips[0].offset;
            RES.fault_byte_offset=flips[0].offset;
            RES.fault_bit        =flips[0].bit;
        }
        // No restore needed – imgBuf is per-run and re-preprocessed next run
    }
    // BUFFERS: injected AFTER inference (see below)

    // ── RUN INFERENCE ─────────────────────────────────────────────────────────
    auto IR=run_inference(runner,
                          imgBuf.data(),inSz,inH,inW,
                          fcBuf.data(),outSz,out_scale,
                          inT[0],outT[0]);

    // ── RESTORE DDR4 BEFORE ANYTHING ELSE (critical!) ─────────────────────────
    // Only reached by WEIGHTS/FEATURE_MAPS/BUFFERS — INSTRUCTIONS returns early.
    if(flip_base && !flips.empty()){
        restore_flips(flip_base,flips);
        if(cfg.verbose)
            sim_log("  [Restore] %zu flips restored in %s\n",
                    flips.size(),flip_tag.c_str());
    }

    // ── HANDLE TIMEOUT / CRASH ────────────────────────────────────────────────
    if(IR.timed_out){
        RES.timeout=true;
        sim_log("[Run %4d] TIMEOUT (DPU hang from %s fault)\n",
                run_id,targetName(eff).c_str());
        // SW-only recovery: recreate runner from original model.
        // hardware_reset_dpu() destroys the XRT bitstream state and breaks
        // all subsequent runs — never use it unless the process is exiting.
        auto nr=recreate_runner(sg);
        if(nr){ runner=nr.release(); RES.recovered=true;
                sim_log("[Run %4d] Runner restored (SW recreate)\n",run_id); }
        else   sim_log("[Run %4d] Runner recreate failed — subsequent runs may fail\n",run_id);
        return true;
    }
    if(IR.exception){
        RES.crash=true;
        sim_log("[Run %4d] EXCEPTION during inference\n",run_id);
        return false;
    }

    // ── POST-INFERENCE FAULT INJECTION (BUFFERS) ──────────────────────────────
    if(eff==FaultTarget::BUFFERS){
        // Flip bits in the DPU output DDR4 buffer (FCResult in main.cc terms)
        uint8_t* out_base=reinterpret_cast<uint8_t*>(fcBuf.data());
        auto post_flips=inject_sbu(out_base,(size_t)outSz,
                                   cfg.k,rng,cfg.verbose,"buffers");
        if(!post_flips.empty()){
            RES.fault_addr       =(uint64_t)out_base+post_flips[0].offset;
            RES.fault_byte_offset=post_flips[0].offset;
            RES.fault_bit        =post_flips[0].bit;
            // Re-run softmax on corrupted fcBuf
            // (run_inference already ran on clean output; redo postprocessing)
            if(!output_tensor_sane(fcBuf.data(),outSz)) RES.output_anomaly=true;
            vector<float> sm(outSz);
            CPUCalcSoftmax(fcBuf.data(),outSz,sm.data(),out_scale);
            if(softmax_anomalous(sm.data(),outSz)) RES.output_anomaly=true;
            auto tk=topk(sm.data(),outSz,TOP_K);
            IR.top1=tk[0]; IR.top1_prob=sm[tk[0]];
            for(int i=0;i<TOP_K;i++) IR.top5[i]=tk[i];
        }
    }

    if(IR.output_bad) RES.output_anomaly=true;

    // ── COMPARE AGAINST GOLDEN ────────────────────────────────────────────────
    RES.faulty_top1      =IR.top1;
    RES.faulty_top1_prob =IR.top1_prob;
    RES.prob_drop        =g_golden.top1_prob-IR.top1_prob;
    RES.top1_match       =(IR.top1==g_golden.top1_class);
    for(int i=0;i<TOP_K;i++)
        if(g_golden.top5_classes[i]==IR.top1){RES.top5_match=true;break;}

    if(cfg.verbose){
        const char* lbl=(IR.top1>=0&&IR.top1<(int)kinds.size())
                        ?kinds[IR.top1].c_str():"?";
        sim_log("[Run %4d] base=%-4d(%.3f) faulty=%-4d(%.3f) %s  DDR4=0x%lX  %s\n",
                run_id,
                g_golden.top1_class,g_golden.top1_prob,
                IR.top1,IR.top1_prob,
                RES.top1_match?"MATCH   ":"MISMATCH",
                RES.fault_addr,lbl);
    }
    return true;
}

// ─────────────────────────────────────────────────────────────────────────────
// STATISTICS
// ─────────────────────────────────────────────────────────────────────────────
struct SimStats {
    int total=0,timeouts=0,crashes=0,anomalies=0,recovered=0;
    int top1_ok=0,top5_ok=0;

    // Coarse accuracy-drop bins (based on top-1 class match)
    int bin_lt1=0,bin_lt2=0,bin_lt5=0,bin_lt10=0,bin_lt20=0,bin_ge20=0;

    // Fine probability degradation bins for TOP-1 class probability
    // Tracks how much the top-1 class probability dropped vs golden baseline,
    // even when the top-1 class identity is preserved (fault masked but prob shifted)
    // Bins: <0.01, <0.05, <0.1, <0.5, <0.75, <1.0, <1.5, <2.0, <3.0, >=3.0  (all in %)
    int pdeg_lt001=0, pdeg_lt005=0, pdeg_lt01=0,  pdeg_lt05=0,
        pdeg_lt075=0, pdeg_lt1=0,   pdeg_lt15=0,  pdeg_lt2=0,
        pdeg_lt3=0,   pdeg_ge3=0;

    map<string,int> t_total,t_miss;
    map<int,int>    TP,FP,FN;
};

static void update_stats(SimStats& S, const RunResult& R){
    S.total++;
    if(R.timeout)        S.timeouts++;
    if(R.crash)          S.crashes++;
    if(R.output_anomaly) S.anomalies++;
    if(R.recovered)      S.recovered++;
    if(R.timeout||R.crash) return;

    if(R.top1_match) S.top1_ok++;
    if(R.top5_match) S.top5_ok++;

    // Coarse accuracy-drop bin (100% drop if top-1 class changed)
    float drop=!R.top1_match?100.f:fabsf(R.prob_drop)*100.f;
    if     (drop< 1.f) S.bin_lt1++;
    else if(drop< 2.f) S.bin_lt2++;
    else if(drop< 5.f) S.bin_lt5++;
    else if(drop<10.f) S.bin_lt10++;
    else if(drop<20.f) S.bin_lt20++;
    else               S.bin_ge20++;

    // Fine probability degradation bin (always based on raw prob_drop)
    // prob_drop = baseline_prob - faulty_prob  (positive = degraded)
    // Use absolute value; negative means prob accidentally increased
    float pd = fabsf(R.prob_drop) * 100.f;  // convert to percentage points
    if     (pd < 0.01f) S.pdeg_lt001++;
    else if(pd < 0.05f) S.pdeg_lt005++;
    else if(pd < 0.10f) S.pdeg_lt01++;
    else if(pd < 0.50f) S.pdeg_lt05++;
    else if(pd < 0.75f) S.pdeg_lt075++;
    else if(pd < 1.00f) S.pdeg_lt1++;
    else if(pd < 1.50f) S.pdeg_lt15++;
    else if(pd < 2.00f) S.pdeg_lt2++;
    else if(pd < 3.00f) S.pdeg_lt3++;
    else                S.pdeg_ge3++;

    S.t_total[R.fault_region]++;
    if(!R.top1_match) S.t_miss[R.fault_region]++;

    if(R.faulty_top1==R.baseline_top1) S.TP[R.faulty_top1]++;
    else{ S.FP[R.faulty_top1]++; S.FN[R.baseline_top1]++; }
}

static void print_report(const SimStats& S, const SimConfig& cfg,
                         const vector<string>& kinds){
    int valid=S.total-S.timeouts-S.crashes;
    auto pct=[&](int n,int d)->float{return d>0?100.f*n/d:0.f;};

    sim_log("\n========================================================\n");
    sim_log("           SBU SIMULATION RESULTS REPORT\n");
    sim_log("========================================================\n");
    sim_log("  Model  : %s\n",cfg.model_path.c_str());
    sim_log("  N=%-4d  k=%-2d  target=%s\n",
            cfg.N,cfg.k,targetName(cfg.target).c_str());
    sim_log("========================================================\n");
    sim_log("  FAULT EVENT SUMMARY\n");
    sim_log("  Total runs          : %5d\n",S.total);
    sim_log("  Timeouts   (DUE)    : %5d  (%5.1f%%)  <- DPU hung from fault\n",
            S.timeouts,pct(S.timeouts,S.total));
    sim_log("  Crashes             : %5d  (%5.1f%%)\n",
            S.crashes,pct(S.crashes,S.total));
    sim_log("  Output anomalies    : %5d  (%5.1f%%)\n",
            S.anomalies,pct(S.anomalies,S.total));
    sim_log("  Recovered           : %5d\n",S.recovered);
    sim_log("  Valid classif. runs : %5d\n",valid);
    sim_log("--------------------------------------------------------\n");
    sim_log("  CLASSIFICATION ACCURACY\n");
    sim_log("  Top-1 preserved  : %5d / %-5d  = %6.2f%%\n",
            S.top1_ok,valid,pct(S.top1_ok,valid));
    sim_log("  Top-5 preserved  : %5d / %-5d  = %6.2f%%\n",
            S.top5_ok,valid,pct(S.top5_ok,valid));
    sim_log("  SDC rate         :                  %6.2f%%\n",
            pct(valid-S.top1_ok,valid));
    sim_log("  Fault masking    :                  %6.2f%%\n",
            pct(S.top1_ok,valid));
    sim_log("--------------------------------------------------------\n");
    sim_log("  ACCURACY-DROP HISTOGRAM  (|prob drop| x 100)\n");
    sim_log("  drop <  1%%  : %5d  (%5.1f%%)  <- masked / negligible\n",
            S.bin_lt1,pct(S.bin_lt1,valid));
    sim_log("  drop <  2%%  : %5d  (%5.1f%%)\n",S.bin_lt2,pct(S.bin_lt2,valid));
    sim_log("  drop <  5%%  : %5d  (%5.1f%%)\n",S.bin_lt5,pct(S.bin_lt5,valid));
    sim_log("  drop < 10%%  : %5d  (%5.1f%%)\n",S.bin_lt10,pct(S.bin_lt10,valid));
    sim_log("  drop < 20%%  : %5d  (%5.1f%%)\n",S.bin_lt20,pct(S.bin_lt20,valid));
    sim_log("  drop >= 20%% : %5d  (%5.1f%%)  <- severe SDC\n",
            S.bin_ge20,pct(S.bin_ge20,valid));
    sim_log("  Negligible-impact (<5%%): %.1f%%\n",
            pct(S.bin_lt1+S.bin_lt2+S.bin_lt5,valid));
    sim_log("--------------------------------------------------------\n");
    sim_log("  TOP-1 PROBABILITY DEGRADATION  (|baseline_prob - faulty_prob| x 100)\n");
    sim_log("  Tracks confidence drop in the top-1 class across all valid runs,\n");
    sim_log("  regardless of whether the class identity changed.\n");
    sim_log("  drop <  0.01pp : %5d  (%5.1f%%)  <- virtually no effect\n",
            S.pdeg_lt001,pct(S.pdeg_lt001,valid));
    sim_log("  drop <  0.05pp : %5d  (%5.1f%%)\n",S.pdeg_lt005,pct(S.pdeg_lt005,valid));
    sim_log("  drop <  0.10pp : %5d  (%5.1f%%)\n",S.pdeg_lt01, pct(S.pdeg_lt01, valid));
    sim_log("  drop <  0.50pp : %5d  (%5.1f%%)\n",S.pdeg_lt05, pct(S.pdeg_lt05, valid));
    sim_log("  drop <  0.75pp : %5d  (%5.1f%%)\n",S.pdeg_lt075,pct(S.pdeg_lt075,valid));
    sim_log("  drop <  1.00pp : %5d  (%5.1f%%)\n",S.pdeg_lt1,  pct(S.pdeg_lt1,  valid));
    sim_log("  drop <  1.50pp : %5d  (%5.1f%%)\n",S.pdeg_lt15, pct(S.pdeg_lt15, valid));
    sim_log("  drop <  2.00pp : %5d  (%5.1f%%)\n",S.pdeg_lt2,  pct(S.pdeg_lt2,  valid));
    sim_log("  drop <  3.00pp : %5d  (%5.1f%%)\n",S.pdeg_lt3,  pct(S.pdeg_lt3,  valid));
    sim_log("  drop >= 3.00pp : %5d  (%5.1f%%)  <- severe confidence loss\n",
            S.pdeg_ge3,  pct(S.pdeg_ge3,  valid));
    sim_log("  (pp = percentage points of softmax probability)\n");
    sim_log("--------------------------------------------------------\n");
    sim_log("  PER-TARGET FAULT SENSITIVITY\n");
    for(auto& [tgt,tot]:S.t_total){
        int miss=S.t_miss.count(tgt)?S.t_miss.at(tgt):0;
        sim_log("  %-22s : %4d runs  %4d mismatches  (%5.1f%%)\n",
                tgt.c_str(),tot,miss,pct(miss,tot));
    }
    sim_log("--------------------------------------------------------\n");
    sim_log("  PRECISION / RECALL  (top-10 active classes)\n");
    sim_log("  %5s  %-24s %4s %4s %4s  %5s  %5s\n",
            "Class","Label","TP","FP","FN","Prec","Recall");
    set<int> classes;
    for(auto& kv:S.TP)classes.insert(kv.first);
    for(auto& kv:S.FP)classes.insert(kv.first);
    for(auto& kv:S.FN)classes.insert(kv.first);
    vector<pair<int,int>> act;
    for(int c:classes){
        int a=(S.TP.count(c)?S.TP.at(c):0)
             +(S.FP.count(c)?S.FP.at(c):0)
             +(S.FN.count(c)?S.FN.at(c):0);
        act.push_back({a,c});
    }
    sort(act.rbegin(),act.rend());
    double psum=0,rsum=0; int pn=0,rn=0,shown=0;
    for(auto& [a,c]:act){
        if(shown++>=10)break;
        int tp=S.TP.count(c)?S.TP.at(c):0;
        int fp=S.FP.count(c)?S.FP.at(c):0;
        int fn=S.FN.count(c)?S.FN.at(c):0;
        float pr=(tp+fp)?100.f*tp/(tp+fp):0.f;
        float re=(tp+fn)?100.f*tp/(tp+fn):0.f;
        string lbl=(c<(int)kinds.size())?kinds[c].substr(0,24):"?";
        sim_log("  %5d  %-24s %4d %4d %4d  %5.1f  %5.1f\n",
                c,lbl.c_str(),tp,fp,fn,pr,re);
        if(tp+fp>0){psum+=pr;pn++;}
        if(tp+fn>0){rsum+=re;rn++;}
    }
    sim_log("  Macro-avg Precision : %.2f%%\n",pn?psum/pn:0.0);
    sim_log("  Macro-avg Recall    : %.2f%%\n",rn?rsum/rn:0.0);
    sim_log("========================================================\n");
}

// ─────────────────────────────────────────────────────────────────────────────
// CSV
// ─────────────────────────────────────────────────────────────────────────────
static void write_csv(const vector<RunResult>& results, const string& path){
    ofstream f(path);
    if(!f){fprintf(stderr,"[CSV] Cannot write %s\n",path.c_str());return;}
    f<<"run_id,target,k,fault_ddr4_addr,fault_byte_offset,fault_bit,fault_region,"
       "timeout,crash,output_anomaly,top1_match,top5_match,"
       "baseline_top1,faulty_top1,baseline_prob,faulty_prob,prob_drop,recovered\n";
    for(auto& R:results){
        f<<R.run_id<<","<<targetName(R.target_used)<<","<<R.bits_flipped<<","
         <<"0x"<<hex<<R.fault_addr<<dec<<","
         <<R.fault_byte_offset<<","<<R.fault_bit<<","<<R.fault_region<<","
         <<R.timeout<<","<<R.crash<<","<<R.output_anomaly<<","
         <<R.top1_match<<","<<R.top5_match<<","
         <<R.baseline_top1<<","<<R.faulty_top1<<","
         <<fixed<<setprecision(6)
         <<R.baseline_top1_prob<<","<<R.faulty_top1_prob<<","
         <<R.prob_drop<<","<<R.recovered<<"\n";
    }
    printf("[CSV] Results -> %s\n",path.c_str());
}

// ─────────────────────────────────────────────────────────────────────────────
// UI
// ─────────────────────────────────────────────────────────────────────────────
static FaultTarget parse_target(const string& s){
    string lo=s; transform(lo.begin(),lo.end(),lo.begin(),::tolower);
    if(lo=="instructions")                         return FaultTarget::INSTRUCTIONS;
    if(lo=="weights")                              return FaultTarget::WEIGHTS;
    if(lo=="feature_maps"||lo=="featuremaps"||lo=="input")
                                                   return FaultTarget::FEATURE_MAPS;
    if(lo=="buffers"||lo=="output")                return FaultTarget::BUFFERS;
    if(lo=="all")                                  return FaultTarget::ALL;
    fprintf(stderr,"[Config] Unknown target '%s', using feature_maps\n",s.c_str());
    return FaultTarget::FEATURE_MAPS;
}

static SimConfig prompt_user(const string& model_path){
    SimConfig cfg; cfg.model_path=model_path;
    printf("\n========================================================\n");
    printf("   SBU Simulation  --  Setup\n");
    printf("========================================================\n\n");
    printf("  Targets (DDR4 regions accessed via VART/XIR API):\n");
    printf("    instructions  - DPU mc_code DDR4 region\n");
    printf("    weights       - DPU parameter DDR4 regions\n");
    printf("    feature_maps  - Input tensor DDR4 buffer\n");
    printf("    buffers       - Output tensor DDR4 buffer\n");
    printf("    all           - random per run\n\n");
    auto ask=[](const char* q, const string& def)->string{
        printf("%s [%s]: ",q,def.c_str());fflush(stdout);
        string line;getline(cin,line);return line.empty()?def:line;};
    cfg.N      =stoi(ask("N (number of runs)","100"));
    cfg.k      =max(1,stoi(ask("k (bit flips per run, 1=SBU)","1")));
    cfg.target =parse_target(ask("target","feature_maps"));
    cfg.verbose=(ask("verbose? (y/N)","n")[0]=='y');
    cfg.output_csv=ask("CSV output","sbu_results.csv");
    printf("\n[Config] N=%d  k=%d  target=%s  verbose=%s\n\n",
           cfg.N,cfg.k,targetName(cfg.target).c_str(),cfg.verbose?"yes":"no");
    return cfg;
}

// ─────────────────────────────────────────────────────────────────────────────
// MAIN
// ─────────────────────────────────────────────────────────────────────────────
int main(int argc, char* argv[]){
    if(argc<2){
        printf("Usage: %s <model.xmodel> [N] [k] [target] [-v]\n",argv[0]);
        printf("  target: instructions|weights|feature_maps|buffers|all\n");
        return -1;
    }

    mt19937 rng(static_cast<uint32_t>(time(nullptr))^(uint32_t)getpid());

    // Install signal handlers for SIGSEGV/SIGABRT — used to catch DPU driver
    // crashes when loading corrupted instructions (INSTRUCTIONS fault target).
    install_crash_handlers();

    SimConfig cfg; cfg.model_path=argv[1];
    if(argc>=5){
        cfg.N      =atoi(argv[2]);
        cfg.k      =max(1,atoi(argv[3]));
        cfg.target =parse_target(argv[4]);
        cfg.verbose=(argc>=6&&string(argv[5])=="-v");
        printf("[Config] N=%d  k=%d  target=%s  verbose=%s\n",
               cfg.N,cfg.k,targetName(cfg.target).c_str(),cfg.verbose?"yes":"no");
    } else {
        cfg=prompt_user(cfg.model_path);
    }

    g_logfp=fopen(cfg.log_file.c_str(),"w");
    if(!g_logfp) fprintf(stderr,"[Warn] Cannot open %s\n",cfg.log_file.c_str());

    vector<string> images,kinds;
    ListImages(baseImagePath,images);
    if(images.empty()){fprintf(stderr,"[Error] No images in %s\n",baseImagePath.c_str());return -1;}
    LoadWords(wordsPath+"words.txt",kinds);
    sim_log("[Setup] %zu images  |  %zu classes\n",images.size(),kinds.size());

    // Load model + runner
    auto graph    =xir::Graph::deserialize(cfg.model_path);
    auto subgraph =get_dpu_subgraph(graph.get());
    CHECK_EQ(subgraph.size(),1u)<<"Expected one DPU subgraph";
    sim_log("[Setup] Subgraph: %s\n",subgraph[0]->get_name().c_str());

    auto runner_owned=vart::Runner::create_runner(subgraph[0],"run");
    vart::Runner* runner=runner_owned.get();

    auto inT =runner->get_input_tensors();
    auto outT=runner->get_output_tensors();
    TensorShape insh[inT.size()],outsh[outT.size()];
    shapes.inTensorList=insh; shapes.outTensorList=outsh;
    getTensorShape(runner,&shapes,(int)inT.size(),(int)outT.size());

    sim_log("[Setup] Input  %s  size=%d h=%d w=%d\n",
            inT[0]->get_name().c_str(),
            shapes.inTensorList[0].size,
            shapes.inTensorList[0].height,
            shapes.inTensorList[0].width);
    sim_log("[Setup] Output %s  size=%d\n",
            outT[0]->get_name().c_str(),
            shapes.outTensorList[0].size);

    // Load clean model bytes once — used for per-run xmodel corruption (INSTRUCTIONS)
    vector<uint8_t> clean_model;
    {
        ifstream mf(cfg.model_path, ios::binary);
        if(!mf){ fprintf(stderr,"[Error] Cannot open model: %s\n",cfg.model_path.c_str()); return -1; }
        clean_model.assign((istreambuf_iterator<char>(mf)), istreambuf_iterator<char>());
        sim_log("[Setup] Model loaded into RAM: %zu bytes  (XMODEL_SAFE_SKIP=%d)\n",
                clean_model.size(), XMODEL_SAFE_SKIP);
        if(clean_model.size() <= (size_t)XMODEL_SAFE_SKIP){
            fprintf(stderr,"[Error] Model smaller than XMODEL_SAFE_SKIP\n"); return -1; }
    }

    // Then resolve and print the specific fault injection regions
    sim_log("[DDR4 Map] Resolving fault injection regions...\n");
    { auto h=get_instruction_region(subgraph[0]);
      if(!h.ptr) sim_log("[DDR4 Map] WARNING: instruction region unavailable\n"); }
    log_weight_regions(subgraph[0]);
    sim_log("[DDR4 Map] Input  buffer: size=%d bytes (addr allocated per-run)\n",
            shapes.inTensorList[0].size);
    sim_log("[DDR4 Map] Output buffer: size=%d bytes (addr allocated per-run)\n",
            shapes.outTensorList[0].size);

    // Cache file offsets for INSTRUCTIONS and WEIGHTS (once — avoids 25MB copy per run)
    cache_region_offsets(subgraph[0], clean_model);

    // Baseline
    if(!run_golden_baseline(runner,images,kinds)){
        fprintf(stderr,"[Error] Baseline failed.\n"); return -1;}

    // Simulation loop
    sim_log("\n[Sim] Starting %d runs  k=%d  target=%s\n\n",
            cfg.N,cfg.k,targetName(cfg.target).c_str());

    SimStats         stats;
    vector<RunResult>all_results;
    all_results.reserve(cfg.N);
    int step=max(1,cfg.N/20);

    for(int run=0;run<cfg.N;run++){
        if(!cfg.verbose&&run%step==0){
            printf("\r[Progress] %4d / %d  (%3.0f%%)   ",
                   run,cfg.N,100.f*run/cfg.N);
            fflush(stdout);
        }
        RunResult R;
        bool ok=perform_faulty_run(runner,subgraph[0],clean_model,images,kinds,
                                   cfg,run,rng,R);
        if(!ok&&!R.timeout){
            sim_log("[Main] Hard crash run %d\n",run);
            auto nr=recreate_runner(subgraph[0]);
            if(nr){runner=nr.release();R.recovered=true;}
            else  sim_log("[Main] Runner recreate failed\n");
        }
        update_stats(stats,R);
        all_results.push_back(R);
    }
    printf("\r[Progress] %4d / %d  (100%%)   \n",cfg.N,cfg.N);

    print_report(stats,cfg,kinds);
    write_csv(all_results,cfg.output_csv);

    if(g_logfp) fclose(g_logfp);
    printf("\n[Done] Log -> %s\n",cfg.log_file.c_str());
    return 0;
}
