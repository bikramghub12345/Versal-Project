/*
 * SBU_simulate.cc  -  Single Bit Upset (SBU) Simulation
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
 * so it sees the corrupted data -- no file reload needed.
 *
 * For INSTRUCTIONS: a single bit flip in mc_code typically causes
 * a DPU hang or exception (caught by timeout + HW reset, Solution 1+2).
 * For WEIGHTS: bit flips in weight data cause wrong conv outputs (SDC).
 * After each run we RESTORE the original bytes so the next run is clean.
 *
 * RECOVERY PATH FOR INSTRUCTIONS (ZCU104 / DPUCZDX8G)
 * =====================================================
 * Corrupted instructions lock the DPU IP core FSM (hard SEFI).
 * The only reliable recovery is a full hardware reset of the PL domain.
 *
 * Recovery sequence (tried in order until one succeeds):
 *   [A] DPU IP soft-reset via AP_CTRL register (0x80000000 / 0x8F000000)
 *       Write 0x0 to AP_CTRL, wait 50ms, write 0x1 to re-enable AP_START
 *   [B] ZynqMP PS->PL reset via CRL_APB (0xFF5E0218) -- asserts pl_resetn0
 *   [C] Sysfs reset:  /sys/class/dpu/dpu0/reset
 *   [D] Sysfs reset:  /sys/bus/platform/drivers/zocl/reset
 *   [E] Bitstream reload via fpgautil / xmutil loadapp (last resort)
 *   [F] Kernel module reload: rmmod xrt_core / modprobe xrt_core
 * After any reset succeeds: exponential-backoff sleep + recreate_runner(sg).
 * Up to MAX_RECOVERY_ATTEMPTS retries before giving up.
 *
 * PROTECTION SOLUTIONS (all except checksum):
 *   [1] Timeout Detection   - kills hanging DPU inference
 *   [2] System / HW Reset   - recovers frozen DPU
 *   [3] Subgraph Tracking   - monitors DPU execution progress
 *   [4] Output Sanity Check - NaN / Inf / all-zero detection
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

// -----------------------------------------------------------------------------
// SIGNAL HANDLER -- catches SIGSEGV/SIGABRT from DPU driver crashes
// -----------------------------------------------------------------------------
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

// -----------------------------------------------------------------------------
// CONSTANTS
// -----------------------------------------------------------------------------
#define INFERENCE_TIMEOUT_MS   10000
#define MAX_RECOVERY_ATTEMPTS  5
#define TOP_K                  5

// XLNX_DPU_TIMEOUT: sets VART's internal DPU watchdog (seconds).
// The child process sets this so VART aborts itself after 10s of DPU hang
// instead of waiting indefinitely. Parent waits INFERENCE_TIMEOUT_MS + 15s margin.
#define CHILD_DPU_TIMEOUT_STR  "10"

// For INSTRUCTIONS fault injection: corrupt xmodel file -> reload -> create_runner()
// copies corrupted instructions into DPU on-chip SRAM.
// 64KB skip gives a safe margin -- random flips land in weight/instruction data,
// not in protobuf structural bytes.
#define XMODEL_SAFE_SKIP       65536   // 64 KB

static const string baseImagePath       = "../images/";
static const string wordsPath           = "./";
static const string CORRUPTED_MODEL_PATH= "/tmp/sbu_corrupted.xmodel";

// -----------------------------------------------------------------------------
// FAULT TARGET
// -----------------------------------------------------------------------------
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

// -----------------------------------------------------------------------------
// CONFIG
// -----------------------------------------------------------------------------
struct SimConfig {
    string      model_path;
    int         N          = 100;
    int         k          = 1;
    FaultTarget target     = FaultTarget::FEATURE_MAPS;
    bool        verbose    = false;
    string      output_csv = "sbu_results.csv";
    string      log_file   = "sbu_sim.log";
};

// -----------------------------------------------------------------------------
// PER-RUN RESULT
// -----------------------------------------------------------------------------
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

// -----------------------------------------------------------------------------
// GLOBAL
// -----------------------------------------------------------------------------
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

// -----------------------------------------------------------------------------
// LOGGING
// -----------------------------------------------------------------------------
static FILE* g_logfp = nullptr;
static void sim_log(const char* fmt, ...) {
    va_list a1,a2;
    va_start(a1,fmt); vprintf(fmt,a1);       va_end(a1);
    if(g_logfp){ va_start(a2,fmt); vfprintf(g_logfp,fmt,a2); va_end(a2); fflush(g_logfp); }
}

// -----------------------------------------------------------------------------
// SOLUTION 1 - TIMEOUT
// -----------------------------------------------------------------------------
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

// -----------------------------------------------------------------------------
// SOLUTION 2 - HW/SW RESET
// -----------------------------------------------------------------------------
/*
 * hardware_reset_dpu() -- multi-path DPU recovery for ZCU104 / DPUCZDX8G
 * ========================================================================
 *
 * When a corrupted instruction stream locks the DPU IP core FSM (hard SEFI),
 * a software-only runner recreate is insufficient -- the IP is physically hung.
 * We must reset the DPU at the hardware level before any VART call will work.
 *
 * Strategy (tried in order; first success wins):
 *
 *   [A] DPU AP_CTRL soft-reset via /dev/mem
 *       Works when the AXI bus to the DPU is still responsive.
 *       Write 0x0 to AP_CTRL (de-assert ap_start), sleep, write 0x1.
 *       DPU base addresses tried: 0x8F000000 (default), 0x80000000 (alt),
 *       0x8FF00000 (custom overlays).
 *       AP_CTRL register is at offset 0x0 (PG338 Table 2-1).
 *
 *   [B] ZynqMP PS->PL reset via CRL_APB (0xFF5E0000 + offset 0x218)
 *       Asserts pl_resetn0 -- resets the entire PL domain.
 *       Safe to use when zocl is the only PL consumer (normal ZCU104 setup).
 *       Read-modify-write: set bit 0 (assert reset), wait 100ms, clear bit 0.
 *
 *   [C] Sysfs DPU reset:  /sys/class/dpu/dpu0/reset  (VART 2.x sysfs node)
 *
 *   [D] Sysfs zocl reset: /sys/bus/platform/drivers/zocl/<device>/reset
 *       Iterates /sys/bus/platform/drivers/zocl/ looking for a "reset" file.
 *
 *   [E] fpgautil / xmutil bitstream reload (last resort; slowest ~3s):
 *       system("fpgautil -b /lib/firmware/xilinx/base/base.bit.bin")
 *       system("xmutil loadapp kv260-dp")    (for KV260 variants)
 *
 *   [F] Kernel module reload: rmmod + modprobe xrt_core (slowest; ~5s)
 *
 * Returns 0 if any method succeeded, -1 if all failed.
 */
static int hardware_reset_dpu() {
    sim_log("[Reset][2-HW] -- Starting DPU hardware reset sequence --\n");

    // -- [A] DPU IP AP_CTRL soft-reset via /dev/mem ------------------------
    {
        // ZCU104 + DPUCZDX8G known DPU AXI-lite base addresses
        static const uint32_t dpu_bases[] = {
            0x8F000000,  // DPUCZDX8G default (ZCU104, ZCU102, Ultra96)
            0x80000000,  // alternate / older board configs
            0x8FF00000,  // custom overlays
        };
        // PG338 Table 2-1: AP_CTRL is at offset 0x00
        static const uint32_t AP_CTRL_OFFSET = 0x00;

        int fd = open("/dev/mem", O_RDWR | O_SYNC);
        if (fd >= 0) {
            for (uint32_t base : dpu_bases) {
                void* b = mmap(NULL, 4096, PROT_READ|PROT_WRITE, MAP_SHARED, fd, (off_t)base);
                if (b == MAP_FAILED) continue;
                volatile uint32_t* ap = (volatile uint32_t*)((volatile uint8_t*)b + AP_CTRL_OFFSET);
                // De-assert ap_start (clear bit 0)
                *ap = 0x00;
                __sync_synchronize();
                usleep(50000);   // 50 ms
                // Re-enable: set ap_start=1 (bit 0)
                *ap = 0x01;
                __sync_synchronize();
                usleep(100000);  // 100 ms let IP come out of reset
                munmap(b, 4096);
                sim_log("[Reset][2-A] AP_CTRL soft-reset at base 0x%08X OK\n", base);
                close(fd);
                goto method_a_done;
            }
            close(fd);
        } else {
            sim_log("[Reset][2-A] Cannot open /dev/mem (%s)\n", strerror(errno));
        }
        sim_log("[Reset][2-A] AP_CTRL soft-reset: no base address responded\n");
        method_a_done:;
    }

    // -- [B] ZynqMP PS->PL hard reset via CRL_APB --------------------------
    {
        // CRL_APB base + PL0_RESET_CTRL offset (ZynqMP TRM, Table 6-96)
        static const off_t  CRL_APB_BASE      = 0xFF5E0000;
        static const uint32_t PL_RESET_OFFSET = 0x218;  // PL0_RESET_CTRL

        int fd = open("/dev/mem", O_RDWR | O_SYNC);
        if (fd >= 0) {
            void* b = mmap(NULL, 4096, PROT_READ|PROT_WRITE, MAP_SHARED, fd, CRL_APB_BASE);
            if (b != MAP_FAILED) {
                volatile uint32_t* rst = (volatile uint32_t*)((volatile uint8_t*)b + PL_RESET_OFFSET);
                uint32_t orig = *rst;
                // Assert reset (bit 0 = 1)
                *rst = orig | 0x1u;
                __sync_synchronize();
                usleep(100000);  // 100 ms -- hold reset
                // De-assert reset (bit 0 = 0)
                *rst = orig & ~0x1u;
                __sync_synchronize();
                usleep(200000);  // 200 ms -- PL boot time
                munmap(b, 4096);
                close(fd);
                sim_log("[Reset][2-B] ZynqMP PL reset (CRL_APB 0xFF5E0218) OK\n");
                return 0;
            }
            close(fd);
        }
        sim_log("[Reset][2-B] ZynqMP CRL_APB mmap failed\n");
    }

    // -- [C] Sysfs DPU reset node ------------------------------------------
    {
        const char* sysfs_paths[] = {
            "/sys/class/dpu/dpu0/reset",
            "/sys/class/dpu/dpu1/reset",
            nullptr
        };
        for (int i = 0; sysfs_paths[i]; i++) {
            FILE* fp = fopen(sysfs_paths[i], "w");
            if (fp) {
                fprintf(fp, "1\n");
                fclose(fp);
                sleep(1);
                sim_log("[Reset][2-C] Sysfs reset via %s OK\n", sysfs_paths[i]);
                return 0;
            }
        }
        sim_log("[Reset][2-C] Sysfs DPU reset nodes not found\n");
    }

    // -- [D] zocl platform driver reset -----------------------------------
    {
        // Walk /sys/bus/platform/drivers/zocl/ for a reset file
        const char* zocl_drv = "/sys/bus/platform/drivers/zocl";
        DIR* dh = opendir(zocl_drv);
        if (dh) {
            struct dirent* de;
            while ((de = readdir(dh))) {
                if (de->d_name[0] == '.') continue;
                char path[512];
                snprintf(path, sizeof(path), "%s/%s/reset", zocl_drv, de->d_name);
                FILE* fp = fopen(path, "w");
                if (fp) {
                    fprintf(fp, "1\n");
                    fclose(fp);
                    closedir(dh);
                    sleep(1);
                    sim_log("[Reset][2-D] zocl driver reset via %s OK\n", path);
                    return 0;
                }
            }
            closedir(dh);
        }
        sim_log("[Reset][2-D] zocl sysfs reset: driver dir not found or no reset node\n");
    }

    // -- [E] fpgautil / xmutil bitstream reload ----------------------------
    {
        // Common firmware paths on ZCU104 / KV260 / ZCU102 PetaLinux images
        static const char* cmds[] = {
            "fpgautil -b /lib/firmware/xilinx/base/base.bit.bin 2>/dev/null",
            "xmutil loadapp kv260-dp 2>/dev/null",
            "fpgautil -b /lib/firmware/base.bit 2>/dev/null",
            nullptr
        };
        for (int i = 0; cmds[i]; i++) {
            if (system(cmds[i]) == 0) {
                sleep(3);  // PL reconfiguration time
                sim_log("[Reset][2-E] Bitstream reload via '%s' OK\n", cmds[i]);
                return 0;
            }
        }
        sim_log("[Reset][2-E] Bitstream reload: all commands failed\n");
    }

    // -- [F] Kernel module reload (last resort) ----------------------------
    {
        if (system("rmmod xrt_core 2>/dev/null; sleep 1; modprobe xrt_core 2>/dev/null") == 0) {
            sleep(2);
            sim_log("[Reset][2-F] Kernel module reload OK\n");
            return 0;
        }
        sim_log("[Reset][2-F] Module reload failed\n");
    }

    sim_log("[Reset][2-HW] !! All reset methods exhausted -- DPU may remain hung !!\n");
    return -1;
}

// -----------------------------------------------------------------------------
// SOLUTION 2 - SW RUNNER RECREATE
// -----------------------------------------------------------------------------
static unique_ptr<vart::Runner> recreate_runner(const xir::Subgraph* sg){
    sim_log("[Reset][2-SW] Recreating runner...\n");
    try{
        auto r=vart::Runner::create_runner(sg,"run");
        sim_log("[Reset][2-SW] Runner recreated OK\n"); return r;
    }catch(const exception& e){sim_log("[Reset][2-SW] FAIL: %s\n",e.what());}
     catch(...){sim_log("[Reset][2-SW] FAIL: unknown\n");}
    return nullptr;
}

// -----------------------------------------------------------------------------
// SOLUTION 2 - FULL RECOVERY AFTER DPU HANG  (hardware + software, with retry)
// -----------------------------------------------------------------------------
/*
 * recover_dpu_after_hang()
 * ========================
 * Called after a confirmed DPU hang (instruction SEFI or timeout from any
 * fault target). Combines hardware reset + runner recreate with exponential
 * backoff and up to MAX_RECOVERY_ATTEMPTS retries.
 *
 * Why retries matter on ZCU104:
 *   - The PL domain takes variable time to come back after CRL_APB reset
 *     depending on bitstream complexity and PS clock gating.
 *   - zocl's XRT context may need extra time to re-enumerate AXI devices.
 *   - First create_runner() call may fail if the DPU /dev/dpu* node
 *     re-appears asynchronously (udev re-scanning takes 100-800 ms).
 *
 * Backoff schedule (attempt 0..MAX_RECOVERY_ATTEMPTS-1):
 *   wait_ms = 500 * 2^attempt  ->  500, 1000, 2000, 4000, 8000 ms
 *
 * Returns true  if runner was successfully recreated (caller should use it).
 * Returns false if all attempts failed (subsequent runs will likely also fail;
 *               the outer loop continues anyway and will collect DUE counts).
 */
static bool recover_dpu_after_hang(vart::Runner*& runner,
                                   const xir::Subgraph* sg,
                                   int run_id)
{
    sim_log("[Recover][run %d] -- DPU hang recovery sequence starting --\n", run_id);

    for (int attempt = 0; attempt < MAX_RECOVERY_ATTEMPTS; attempt++) {
        sim_log("[Recover][%d/%d] Attempt %d of %d\n",
                attempt+1, MAX_RECOVERY_ATTEMPTS, attempt+1, MAX_RECOVERY_ATTEMPTS);

        // Step 1: Hardware reset
        int hw_ret = hardware_reset_dpu();
        if (hw_ret != 0) {
            sim_log("[Recover][%d/%d] Hardware reset returned non-zero; "
                    "trying runner recreate anyway\n", attempt+1, MAX_RECOVERY_ATTEMPTS);
        }

        // Step 2: Exponential-backoff sleep -- give PL / zocl time to recover
        int wait_ms = 500 * (1 << attempt);  // 500, 1000, 2000, 4000, 8000 ms
        sim_log("[Recover][%d/%d] Waiting %d ms for DPU/zocl to stabilize...\n",
                attempt+1, MAX_RECOVERY_ATTEMPTS, wait_ms);
        this_thread::sleep_for(chrono::milliseconds(wait_ms));

        // Step 3: Reinstall signal handlers (hardware_reset may have triggered some)
        reinstall_crash_handlers();

        // Step 4: Try to recreate the runner
        auto nr = recreate_runner(sg);
        if (nr) {
            runner = nr.release();
            sim_log("[Recover][%d/%d] SUCCESS -- DPU recovered and runner is live\n",
                    attempt+1, MAX_RECOVERY_ATTEMPTS);
            return true;
        }

        sim_log("[Recover][%d/%d] Runner recreate failed -- will retry\n",
                attempt+1, MAX_RECOVERY_ATTEMPTS);
    }

    sim_log("[Recover] !! FAILED after %d attempts -- DPU remains hung !!\n",
            MAX_RECOVERY_ATTEMPTS);
    sim_log("[Recover] Subsequent runs will continue collecting DUE statistics.\n");
    return false;
}

// -----------------------------------------------------------------------------
// SOLUTION 4 - OUTPUT SANITY
// -----------------------------------------------------------------------------
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

// -----------------------------------------------------------------------------
// PREPROCESSING / POSTPROCESSING
// -----------------------------------------------------------------------------
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

// -----------------------------------------------------------------------------
// FILE HELPERS
// -----------------------------------------------------------------------------
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

// -----------------------------------------------------------------------------
// SBU FLIP PRIMITIVE  (operates on a raw byte pointer)
// -----------------------------------------------------------------------------
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

// -----------------------------------------------------------------------------
// DDR4 REGION ACCESS VIA XIR SUBGRAPH ATTRIBUTES
// -----------------------------------------------------------------------------
struct RegionHandle {
    uint8_t* ptr  = nullptr;
    size_t   size = 0;
    string   name;
};

static void dump_subgraph_attrs(const xir::Subgraph* sg) {
    sim_log("\n[AttrDump] Subgraph: %s\n", sg->get_name().c_str());
    sim_log("[AttrDump] Probing known XIR attributes:\n");

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

static RegionHandle get_instruction_region(const xir::Subgraph* sg) {
    RegionHandle h;

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
    return h;
}

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

static size_t find_bytes_offset(const vector<uint8_t>& haystack,
                                 const vector<char>& needle,
                                 size_t search_start = 0) {
    if(needle.empty() || needle.size() > haystack.size()) return string::npos;
    size_t sig_len = min((size_t)32, needle.size());
    const uint8_t* h = haystack.data() + search_start;
    size_t h_len = haystack.size() - search_start;
    for(size_t i = 0; i + sig_len <= h_len; i++){
        if(memcmp(h + i, needle.data(), sig_len) == 0){
            size_t verify_len = min((size_t)256, needle.size());
            if(memcmp(h + i, needle.data(), verify_len) == 0)
                return search_start + i;
        }
    }
    return string::npos;
}

static size_t g_mc_code_offset = string::npos;
static size_t g_mc_code_size   = 0;
static size_t g_weight_offset  = string::npos;
static size_t g_weight_size    = 0;

static void cache_region_offsets(const xir::Subgraph* sg,
                                  const vector<uint8_t>& clean_model) {
    auto* sgm = const_cast<xir::Subgraph*>(sg);

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

// -----------------------------------------------------------------------------
// INFERENCE HELPER
// -----------------------------------------------------------------------------
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

    g_tracker.reset(1);

    int ret = run_with_timeout(runner, ip, op, INFERENCE_TIMEOUT_MS);

    if(ret==1){ R.timed_out=true;  return R; }
    if(ret==2){ R.exception=true;  return R; }

    g_tracker.mark_complete();

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

// -----------------------------------------------------------------------------
// GOLDEN BASELINE
// -----------------------------------------------------------------------------
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

// -----------------------------------------------------------------------------
// SUBPROCESS INFERENCE  (INSTRUCTIONS and WEIGHTS fault targets)
// -----------------------------------------------------------------------------
/*
 * run_in_subprocess()
 * ===================
 * Isolates VART's abort() from the main process.
 *
 * KEY CHANGES vs original:
 *
 *   1. Child sets XLNX_DPU_TIMEOUT = CHILD_DPU_TIMEOUT_STR (10s).
 *      This activates VART's internal DPU watchdog timer.  When the DPU
 *      hangs on corrupted instructions, VART's timeout thread calls
 *      LOG(FATAL) -> abort() after exactly 10 seconds, killing only the
 *      child.  Without this env var the child hangs indefinitely and the
 *      parent must SIGKILL it manually at INFERENCE_TIMEOUT_MS + 15s.
 *
 *   2. On child death the parent now calls recover_dpu_after_hang()
 *      instead of a single-shot hardware_reset_dpu() + recreate_runner().
 *      recover_dpu_after_hang() tries all 6 hardware reset methods and
 *      retries runner creation up to MAX_RECOVERY_ATTEMPTS times with
 *      exponential backoff.  This is the fix for the DPU remaining hung
 *      after the first recovery attempt fails.
 *
 *   3. The corrupted xmodel file is always unlinked before returning,
 *      whether the child succeeded or not, to prevent stale files
 *      accumulating in /tmp/.
 */
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
    if(pipe(pfd)!=0){
        sim_log("[Run %4d] pipe: %s\n",run_id,strerror(errno));
        RES.crash=true; return false;
    }

    pid_t pid = fork();
    if(pid<0){
        close(pfd[0]); close(pfd[1]);
        sim_log("[Run %4d] fork: %s\n",run_id,strerror(errno));
        RES.crash=true; return false;
    }

    if(pid==0){
        // -- CHILD ---------------------------------------------------------
        close(pfd[0]);
        auto die=[&](){ write(pfd[1],"DUE\n",4); close(pfd[1]); _exit(1); };

        // FIX 1: Set VART's DPU timeout watchdog so the child's VART
        //        aborts itself after CHILD_DPU_TIMEOUT_STR seconds of DPU hang.
        //        This prevents the child from blocking indefinitely and
        //        ensures the DPU IP reset fires at a predictable time.
        setenv("XLNX_DPU_TIMEOUT", CHILD_DPU_TIMEOUT_STR, 1);

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

            // No software timeout in child -- XLNX_DPU_TIMEOUT handles this.
            // If DPU hangs: VART's internal watchdog fires -> LOG(FATAL) -> abort()
            // -> child exits with SIGABRT -> parent detects and calls recover_dpu_after_hang().
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

    // -- PARENT ------------------------------------------------------------
    close(pfd[1]);

    // Wait for child. Timeout = VART's internal timeout + 15s safety margin.
    // The child should self-abort via XLNX_DPU_TIMEOUT after ~10s of DPU hang.
    const int WAIT_MS = INFERENCE_TIMEOUT_MS + 15000;
    auto t0c = chrono::steady_clock::now();
    int wst=0; pid_t wr=0;
    while(true){
        wr=waitpid(pid,&wst,WNOHANG);
        if(wr==pid) break;
        if(chrono::duration_cast<chrono::milliseconds>(
               chrono::steady_clock::now()-t0c).count()>WAIT_MS){
            sim_log("[Run %4d] Child exceeded wait budget (%d ms) -- SIGKILL\n",
                    run_id, WAIT_MS);
            kill(pid,SIGKILL); waitpid(pid,&wst,0); break;
        }
        this_thread::sleep_for(chrono::milliseconds(100));
    }

    // Always read pipe and unlink temp file
    char rbuf[256]={}; read(pfd[0],rbuf,sizeof(rbuf)-1);
    close(pfd[0]);
    unlink(corrupted_path.c_str());

    bool child_ok = (WIFEXITED(wst)&&WEXITSTATUS(wst)==0);
    if(!child_ok){
        // Child died -- DPU hung from corrupted instructions (hard SEFI).
        // The child process exiting frees zocl's ERT execution context,
        // but the DPU IP core FSM is still locked at the hardware level.
        //
        // FIX 2: Call recover_dpu_after_hang() instead of a single-shot
        // hardware_reset_dpu() + recreate_runner(). This retries the full
        // hardware reset + runner creation sequence MAX_RECOVERY_ATTEMPTS
        // times with exponential backoff, giving the PL domain enough time
        // to stabilize before VART tries to re-enumerate the DPU device.
        sim_log("[Run %4d] Subprocess died (exit=%d sig=%d) -- DPU SEFI detected\n",
                run_id,
                WIFEXITED(wst)   ? WEXITSTATUS(wst) : -1,
                WIFSIGNALED(wst) ? WTERMSIG(wst)    : 0);
        sim_log("[Run %4d] Initiating full DPU recovery...\n", run_id);

        bool recovered = recover_dpu_after_hang(runner, sg, run_id);
        RES.recovered = recovered;
        RES.timeout   = true;  // Counts as DUE (Detected Unrecoverable Error)

        if (!recovered) {
            sim_log("[Run %4d] WARNING: DPU recovery failed -- "
                    "subsequent runs may also time out\n", run_id);
        }
        return true;
    }

    // Child exited cleanly -- parse result
    if(strncmp(rbuf,"OK ",3)!=0){
        sim_log("[Run %4d] Child OK exit but bad pipe data: '%s'\n", run_id, rbuf);
        RES.crash=true; return true;
    }
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

// -----------------------------------------------------------------------------
// SINGLE FAULTY RUN
// -----------------------------------------------------------------------------
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

    // -- FAULT INJECTION -------------------------------------------------------
    vector<FlipInfo> flips;
    uint8_t* flip_base = nullptr;
    size_t   flip_size = 0;
    string   flip_tag;

    if(eff==FaultTarget::INSTRUCTIONS){
        // -- File-patch + subprocess for INSTRUCTIONS only --------------------
        // Corrupted instructions lock the DPU FSM (hard SEFI) so we isolate
        // in a child process. Never mix with WEIGHTS -- WEIGHTS do not need
        // a subprocess and the XRT buffer re-allocation inside the child
        // crashes with allocation failure on ZCU104.
        if(clean_model.empty() || g_mc_code_offset==string::npos || g_mc_code_size==0){
            sim_log("[Run %4d] mc_code region not cached -- skipping\n",run_id);
            RES.crash=true; return false;
        }
        vector<uint8_t> patched = clean_model;
        vector<FlipInfo> pf = inject_sbu(
            patched.data() + g_mc_code_offset, g_mc_code_size,
            cfg.k, rng, cfg.verbose, "mc_code");
        if(!pf.empty()){
            RES.fault_byte_offset = g_mc_code_offset + pf[0].offset;
            RES.fault_addr        = RES.fault_byte_offset;
            RES.fault_bit         = pf[0].bit;
        }
        {
            ofstream out(CORRUPTED_MODEL_PATH, ios::binary);
            if(!out){
                sim_log("[Run %4d] Cannot write %s\n",run_id,CORRUPTED_MODEL_PATH.c_str());
                RES.crash=true; return false;
            }
            out.write(reinterpret_cast<const char*>(patched.data()), patched.size());
        }
        return run_in_subprocess(
            CORRUPTED_MODEL_PATH,
            imgBuf.data(), outSz, out_scale,
            run_id, cfg.verbose,
            runner, sg, RES);

    } else if(eff==FaultTarget::WEIGHTS){
        // -- Direct in-memory flip for WEIGHTS --------------------------------
        // Weights live in DDR4 and are already mapped into this process via
        // the XIR subgraph attribute pointer. Flipping them NEVER locks the
        // DPU FSM -- worst case is SDC (wrong conv output). We flip directly,
        // run inference on the existing runner, then restore. No subprocess,
        // no XRT buffer re-allocation, no recovery needed.
        try {
            auto& wmap = const_cast<xir::Subgraph*>(sg)
                         ->get_attr<map<string,vector<char>>>("reg_id_to_parameter_value");
            uint8_t* wptr = nullptr;
            size_t   wsz  = 0;
            string wname;
            for(auto& [rid, d] : wmap){
                if(!d.empty()){
                    wptr  = reinterpret_cast<uint8_t*>(const_cast<char*>(d.data()));
                    wsz   = d.size();
                    wname = rid;
                    break;
                }
            }
            if(!wptr){
                sim_log("[Run %4d] No weight region found\n",run_id);
                RES.crash=true; return false;
            }
            if(cfg.verbose)
                sim_log("  [WeightFlip] region=%s  vaddr=0x%lX  size=%zu\n",
                        wname.c_str(), (uint64_t)wptr, wsz);

            vector<FlipInfo> wf = inject_sbu(wptr, wsz, cfg.k, rng, cfg.verbose, wname.c_str());
            if(!wf.empty()){
                RES.fault_addr       = (uint64_t)wptr + wf[0].offset;
                RES.fault_byte_offset= wf[0].offset;
                RES.fault_bit        = wf[0].bit;
            }

            // Run inference with corrupted weights in DDR4
            auto IR = run_inference(runner,
                                    imgBuf.data(),inSz,inH,inW,
                                    fcBuf.data(),outSz,out_scale,
                                    inT[0],outT[0]);

            // Restore immediately -- before any logging or branching
            if(!wf.empty()) restore_flips(wptr, wf);
            if(cfg.verbose)
                sim_log("  [WeightFlip] %zu flips restored in %s\n",
                        wf.size(), wname.c_str());

            if(IR.timed_out){ RES.timeout=true; return true; }
            if(IR.exception){ RES.crash=true;   return false; }
            if(IR.output_bad) RES.output_anomaly=true;

            RES.faulty_top1      = IR.top1;
            RES.faulty_top1_prob = IR.top1_prob;
            RES.prob_drop        = g_golden.top1_prob - IR.top1_prob;
            RES.top1_match       = (IR.top1 == g_golden.top1_class);
            for(int i=0;i<TOP_K;i++)
                if(g_golden.top5_classes[i]==IR.top1){RES.top5_match=true;break;}
            if(cfg.verbose){
                const char* lbl=(IR.top1>=0&&IR.top1<(int)kinds.size())
                                ?kinds[IR.top1].c_str():"?";
                sim_log("[Run %4d] base=%-4d(%.3f) faulty=%-4d(%.3f) %s  %s\n",
                        run_id, g_golden.top1_class, g_golden.top1_prob,
                        IR.top1, IR.top1_prob,
                        RES.top1_match?"MATCH   ":"MISMATCH", lbl);
            }
            return true;
        } catch(const exception& e){
            sim_log("[Run %4d] Weight attr access failed: %s\n", run_id, e.what());
            RES.crash=true; return false;
        }

    } else if(eff==FaultTarget::FEATURE_MAPS){
        flip_base=reinterpret_cast<uint8_t*>(imgBuf.data());
        flip_size=(size_t)inSz;
        flip_tag ="feature_maps";
        flips=inject_sbu(flip_base,flip_size,cfg.k,rng,cfg.verbose,flip_tag.c_str());
        if(!flips.empty()){
            RES.fault_addr       =(uint64_t)flip_base+flips[0].offset;
            RES.fault_byte_offset=flips[0].offset;
            RES.fault_bit        =flips[0].bit;
        }
    }
    // BUFFERS: injected AFTER inference (see below)

    // -- RUN INFERENCE ---------------------------------------------------------
    auto IR=run_inference(runner,
                          imgBuf.data(),inSz,inH,inW,
                          fcBuf.data(),outSz,out_scale,
                          inT[0],outT[0]);

    // -- RESTORE DDR4 BEFORE ANYTHING ELSE (critical!) -------------------------
    if(flip_base && !flips.empty()){
        restore_flips(flip_base,flips);
        if(cfg.verbose)
            sim_log("  [Restore] %zu flips restored in %s\n",
                    flips.size(),flip_tag.c_str());
    }

    // -- HANDLE TIMEOUT / CRASH ------------------------------------------------
    if(IR.timed_out){
        RES.timeout=true;
        sim_log("[Run %4d] TIMEOUT (DPU hang from %s fault)\n",
                run_id,targetName(eff).c_str());
        // SW-only recovery for non-instruction faults: recreate runner only.
        // hardware_reset_dpu() destroys the XRT bitstream state and breaks
        // all subsequent runs -- only use it for the instruction SEFI case
        // (handled inside run_in_subprocess above).
        auto nr=recreate_runner(sg);
        if(nr){ runner=nr.release(); RES.recovered=true;
                sim_log("[Run %4d] Runner restored (SW recreate)\n",run_id); }
        else   sim_log("[Run %4d] Runner recreate failed\n",run_id);
        return true;
    }
    if(IR.exception){
        RES.crash=true;
        sim_log("[Run %4d] EXCEPTION during inference\n",run_id);
        return false;
    }

    // -- POST-INFERENCE FAULT INJECTION (BUFFERS) ------------------------------
    if(eff==FaultTarget::BUFFERS){
        uint8_t* out_base=reinterpret_cast<uint8_t*>(fcBuf.data());
        auto post_flips=inject_sbu(out_base,(size_t)outSz,
                                   cfg.k,rng,cfg.verbose,"buffers");
        if(!post_flips.empty()){
            RES.fault_addr       =(uint64_t)out_base+post_flips[0].offset;
            RES.fault_byte_offset=post_flips[0].offset;
            RES.fault_bit        =post_flips[0].bit;
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

    // -- COMPARE AGAINST GOLDEN ------------------------------------------------
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

// -----------------------------------------------------------------------------
// STATISTICS
// -----------------------------------------------------------------------------
struct SimStats {
    int total=0,timeouts=0,crashes=0,anomalies=0,recovered=0;
    int top1_ok=0,top5_ok=0;

    int bin_lt1=0,bin_lt2=0,bin_lt5=0,bin_lt10=0,bin_lt20=0,bin_ge20=0;

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

    float drop=!R.top1_match?100.f:fabsf(R.prob_drop)*100.f;
    if     (drop< 1.f) S.bin_lt1++;
    else if(drop< 2.f) S.bin_lt2++;
    else if(drop< 5.f) S.bin_lt5++;
    else if(drop<10.f) S.bin_lt10++;
    else if(drop<20.f) S.bin_lt20++;
    else               S.bin_ge20++;

    float pd = fabsf(R.prob_drop) * 100.f;
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
    sim_log("  drop <  0.75pp : %5d  (%5.1f%%)  \n",S.pdeg_lt075,pct(S.pdeg_lt075,valid));
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

// -----------------------------------------------------------------------------
// CSV
// -----------------------------------------------------------------------------
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

// -----------------------------------------------------------------------------
// UI
// -----------------------------------------------------------------------------
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

// -----------------------------------------------------------------------------
// MAIN
// -----------------------------------------------------------------------------
int main(int argc, char* argv[]){
    if(argc<2){
        printf("Usage: %s <model.xmodel> [N] [k] [target] [-v]\n",argv[0]);
        printf("  target: instructions|weights|feature_maps|buffers|all\n");
        return -1;
    }

    mt19937 rng(static_cast<uint32_t>(time(nullptr))^(uint32_t)getpid());

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

    // Load clean model bytes once -- used for per-run xmodel corruption
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

    // Resolve DDR4 regions
    sim_log("[DDR4 Map] Resolving fault injection regions...\n");
    { auto h=get_instruction_region(subgraph[0]);
      if(!h.ptr) sim_log("[DDR4 Map] WARNING: instruction region unavailable\n"); }
    log_weight_regions(subgraph[0]);
    sim_log("[DDR4 Map] Input  buffer: size=%d bytes (addr allocated per-run)\n",
            shapes.inTensorList[0].size);
    sim_log("[DDR4 Map] Output buffer: size=%d bytes (addr allocated per-run)\n",
            shapes.outTensorList[0].size);

    // Cache file offsets once -- avoids 25MB get_attr copy per run
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
            sim_log("[Main] Hard crash run %d -- attempting full recovery\n",run);
            bool recovered = recover_dpu_after_hang(runner, subgraph[0], run);
            R.recovered = recovered;
            if(!recovered)
                sim_log("[Main] Recovery failed for run %d\n", run);
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
