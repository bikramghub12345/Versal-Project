/*
 * MBU_simulate.cc  –  Multi-Bit Upset (MBU) Fault Injection Simulator
 * ======================================================================
 * Extends SBU_simulate to:
 *   - Accept multiple bit counts (e.g. 1 2 3 5 10)
 *   - Run N simulations per image per bit count
 *   - Accept an image folder (golden = fault-free inference on each image)
 *   - Save per-bit CSV:  results_k<N>_bits.csv
 *   - Save accuracy CSV: accuracy_summary.csv
 *   - Generate plot_results.py for histogram plots
 *
 * Usage:
 *   ./MBU_simulate <model.xmodel> <image_folder> [target] [-v]
 *   (remaining params prompted interactively)
 *
 * BUILD:
 *   g++ -std=c++17 -O2 -o MBU_simulate src/MBU_simulate.cc \
 *       ../common/common.cpp \
 *       -I ./src -I ../common \
 *       -I /usr/include/opencv4 -I /usr/include/vitis_ai \
 *       $(pkg-config --cflags --libs opencv4) \
 *       -lvart-runner -lxir -lglog -lpthread
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
// SIGNAL HANDLER
// -----------------------------------------------------------------------------
static sigjmp_buf  g_crash_jmp;
static volatile sig_atomic_t g_in_protected = 0;

static void crash_signal_handler(int sig) {
    if (g_in_protected) { siglongjmp(g_crash_jmp, sig); }
    ::signal(sig, SIG_DFL); ::raise(sig);
}
static void install_crash_handlers() {
    struct ::sigaction sa; memset(&sa,0,sizeof(sa));
    sa.sa_handler=crash_signal_handler; sigemptyset(&sa.sa_mask);
    sa.sa_flags=SA_RESETHAND;
    ::sigaction(SIGSEGV,&sa,nullptr); ::sigaction(SIGABRT,&sa,nullptr);
    ::sigaction(SIGBUS, &sa,nullptr);
}
static void reinstall_crash_handlers(){ install_crash_handlers(); }

// -----------------------------------------------------------------------------
// CONSTANTS
// -----------------------------------------------------------------------------
#define INFERENCE_TIMEOUT_MS  10000
#define TOP_K                 5
#define XMODEL_SAFE_SKIP      65536

static const string CORRUPTED_MODEL_PATH = "/tmp/mbu_corrupted.xmodel";
static const string wordsPath            = "./";

// -----------------------------------------------------------------------------
// FAULT TARGET
// -----------------------------------------------------------------------------
enum class FaultTarget { INSTRUCTIONS, WEIGHTS, FEATURE_MAPS, BUFFERS, ALL };

static string targetName(FaultTarget t){
    switch(t){
        case FaultTarget::INSTRUCTIONS: return "INSTRUCTIONS";
        case FaultTarget::WEIGHTS:      return "WEIGHTS";
        case FaultTarget::FEATURE_MAPS: return "FEATURE_MAPS";
        case FaultTarget::BUFFERS:      return "BUFFERS";
        case FaultTarget::ALL:          return "ALL";
    }
    return "UNKNOWN";
}

// -----------------------------------------------------------------------------
// BASELINE — clean model output for one image (no fault)
// -----------------------------------------------------------------------------
struct BaselineResult {
    string image_name;
    string image_path;
    int    ground_truth_class = -1;  // from folder/synset — the REAL label
    string ground_truth_name;
    int    baseline_class     = -1;  // clean model top-1 (may differ from ground truth)
    string baseline_name;
    float  baseline_prob      = 0.f;
    bool   valid              = false;
};

// -----------------------------------------------------------------------------
// PER-IMAGE RESULT (one row per image per k in the CSV)
// -----------------------------------------------------------------------------
struct RunResultMBU {
    // Identity
    string      image_name;
    int         k_bits        = 0;
    FaultTarget target_used   = FaultTarget::FEATURE_MAPS;

    // Ground truth (from ImageNet val folder structure)
    int    ground_truth_class = -1;
    string ground_truth_name;

    // Baseline — clean model output (may differ from ground truth)
    int    baseline_class = -1;
    string baseline_name;
    float  baseline_prob  = 0.f;

    // Faulty top-3
    int    faulty_class[3]  = {-1,-1,-1};
    float  faulty_prob[3]   = {0,0,0};
    string faulty_name[3];

    // Derived
    bool  correctly_classified = false;  // faulty_top1 == ground_truth_class
    float prob_drop            = 0.f;    // baseline_prob - faulty_top1_prob

    // Status
    bool  timeout        = false;
    bool  crash          = false;
    bool  output_anomaly = false;
    bool  recovered      = false;

    // Fault location
    uint64_t fault_addr        = 0;
    size_t   fault_byte_offset = 0;
    int      fault_bit         = 0;
};

// -----------------------------------------------------------------------------
// CONFIG
// -----------------------------------------------------------------------------
struct SimConfig {
    string          model_path;
    string          val_folder;
    vector<int>     bit_counts;
    FaultTarget     target      = FaultTarget::FEATURE_MAPS;
    bool            verbose     = false;
    string          base_name   = "mbu_results";  // FaultResults/<base_name>/<target>/
};

// -----------------------------------------------------------------------------
// FOLDER STRUCTURE HELPERS
// FaultResults/<base_name>/<target_name>/
// -----------------------------------------------------------------------------
static string targetDirName(FaultTarget t){
    switch(t){
        case FaultTarget::INSTRUCTIONS: return "instructions";
        case FaultTarget::WEIGHTS:      return "weights";
        case FaultTarget::FEATURE_MAPS: return "feature_maps";
        case FaultTarget::BUFFERS:      return "buffers";
        case FaultTarget::ALL:          return "all";
    }
    return "unknown";
}

// Recursively create directory (mkdir -p equivalent)
static void mkdirp(const string& path){
    string tmp = path;
    for(size_t i=1; i<tmp.size(); i++){
        if(tmp[i]=='/'){
            tmp[i]='\0';
            mkdir(tmp.c_str(), 0755);
            tmp[i]='/';
        }
    }
    mkdir(tmp.c_str(), 0755);
}

// Remove all files in a directory (but keep the directory itself)
static void clear_dir(const string& path){
    DIR* d=opendir(path.c_str());
    if(!d) return;
    struct dirent* e;
    while((e=readdir(d))!=nullptr){
        if(string(e->d_name)=="."||string(e->d_name)=="..") continue;
        string fp=path+"/"+e->d_name;
        struct stat s; lstat(fp.c_str(),&s);
        if(S_ISREG(s.st_mode)) unlink(fp.c_str());
    }
    closedir(d);
}

// Build and prepare the output directory for one target experiment:
//   FaultResults/<base_name>/<target_name>/
// Creates it if not present, clears existing files if already there.
static string prepare_target_dir(const string& base_name, FaultTarget target){
    string root  = "./FaultResults";
    string exp   = root + "/" + base_name;
    string tdir  = exp  + "/" + targetDirName(target);
    mkdirp(tdir);
    clear_dir(tdir);
    printf("[Dir] Output: %s\n", tdir.c_str());
    return tdir;
}
GraphInfo shapes;

static size_t g_mc_code_offset = string::npos;
static size_t g_mc_code_size   = 0;
static size_t g_weight_offset  = string::npos;
static size_t g_weight_size    = 0;

// -----------------------------------------------------------------------------
// LOGGING
// -----------------------------------------------------------------------------
static FILE* g_logfp = nullptr;
static void sim_log(const char* fmt, ...) {
    va_list a1,a2;
    va_start(a1,fmt); vprintf(fmt,a1); va_end(a1);
    if(g_logfp){ va_start(a2,fmt); vfprintf(g_logfp,fmt,a2); va_end(a2); fflush(g_logfp); }
}

// -----------------------------------------------------------------------------
// SOLUTION 1 – TIMEOUT
// -----------------------------------------------------------------------------
static int run_with_timeout(vart::Runner* runner,
                            vector<vart::TensorBuffer*>& ip,
                            vector<vart::TensorBuffer*>& op,
                            int timeout_ms){
    atomic<int> result{-1}; atomic<bool> done{false};
    thread t([&](){ try{
        auto job=runner->execute_async(ip,op);
        result=(runner->wait(job.first,timeout_ms)==0)?0:2;
    }catch(...){result=2;} done=true; });
    auto t0=steady_clock::now();
    while(!done){
        this_thread::sleep_for(milliseconds(50));
        if(duration_cast<milliseconds>(steady_clock::now()-t0).count()>=timeout_ms){
            sim_log("[Timeout] DPU exceeded %d ms\n",timeout_ms);
            result=1; done=true; break;
        }
    }
    t.detach(); return result.load();
}

// -----------------------------------------------------------------------------
// SOLUTION 2 – HW/SW RESET
// -----------------------------------------------------------------------------
static int hardware_reset_dpu(){
    sim_log("[Reset][HW] Attempting hardware DPU reset...\n");
    int fd=open("/dev/mem",O_RDWR|O_SYNC);
    if(fd>=0){
        void* b=mmap(NULL,4096,PROT_READ|PROT_WRITE,MAP_SHARED,fd,0x80000000);
        if(b!=MAP_FAILED){
            volatile uint32_t* c=(volatile uint32_t*)b;
            c[0]=0x00;usleep(10000);c[1]=0xFF;usleep(10000);
            c[0]=0x01;usleep(100000);c[0]=0x04;usleep(10000);
            munmap(b,4096);close(fd);
            sim_log("[Reset][HW] Done\n"); return 0;
        }
        close(fd);
    }
    FILE* fp=fopen("/sys/class/dpu/dpu0/reset","w");
    if(fp){fprintf(fp,"1\n");fclose(fp);sleep(1); return 0;}
    if(system("rmmod xrt_core 2>/dev/null;sleep 1;modprobe xrt_core 2>/dev/null")==0){
        sleep(2); return 0;}
    return -1;
}

static unique_ptr<vart::Runner> recreate_runner(const xir::Subgraph* sg){
    sim_log("[Reset][SW] Recreating runner...\n");
    try{ auto r=vart::Runner::create_runner(sg,"run");
         sim_log("[Reset][SW] OK\n"); return r; }
    catch(const exception& e){sim_log("[Reset][SW] FAIL: %s\n",e.what());}
    catch(...){sim_log("[Reset][SW] FAIL\n");}
    return nullptr;
}

// -----------------------------------------------------------------------------
// SOLUTION 4 – OUTPUT SANITY
// -----------------------------------------------------------------------------
static bool output_tensor_sane(const int8_t* d, int sz){
    if(sz<=0) return false;
    int zeros=0,mn=127,mx=-128;
    for(int i=0;i<sz;i++){
        int v=(int)d[i]; if(v==0)zeros++;
        if(v<mn)mn=v; if(v>mx)mx=v;
    }
    if(zeros==sz||mn==mx) return false;
    return true;
}
static bool softmax_anomalous(const float* s,int sz){
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
// Image entry with ground truth class from folder structure
struct ImageEntry {
    string path;               // full path to image file
    string name;               // filename only
    int    ground_truth = -1;  // class index from synset folder
};

// Load synset.txt — 1000 lines, one synset ID per line (e.g. n01440764)
// Line index 0..999 matches words.txt class 0..999
static map<string,int> LoadSynsets(const string& path){
    map<string,int> m;
    ifstream f(path);
    if(!f){ fprintf(stderr,"[Warn] synset.txt not found at %s\n",path.c_str()); return m; }
    string line; int idx=0;
    while(getline(f,line)){
        if(!line.empty()) m[line]=idx;
        idx++;
    }
    return m;
}

// Walk val/ subdirectories: val/n01440764/*.JPEG
// Each subdir name = synset ID → looked up in synset_to_idx for class index
static void ListImagesWithGroundTruth(const string& val_dir,
                                       const map<string,int>& synset_to_idx,
                                       vector<ImageEntry>& entries){
    entries.clear();
    struct stat s; lstat(val_dir.c_str(),&s);
    if(!S_ISDIR(s.st_mode)){
        fprintf(stderr,"[Error] %s is not a directory\n",val_dir.c_str()); exit(1);
    }
    DIR* top=opendir(val_dir.c_str());
    if(!top){ fprintf(stderr,"[Error] Cannot open %s\n",val_dir.c_str()); exit(1); }
    struct dirent* cls_entry;
    while((cls_entry=readdir(top))!=nullptr){
        if(cls_entry->d_name[0]=='.') continue;
        string synset=cls_entry->d_name;
        string cls_path=val_dir+"/"+synset;
        struct stat cs; lstat(cls_path.c_str(),&cs);
        if(!S_ISDIR(cs.st_mode)) continue;

        // Lookup ground truth class index
        int gt_class=-1;
        auto it=synset_to_idx.find(synset);
        if(it!=synset_to_idx.end()) gt_class=it->second;
        else{
            fprintf(stderr,"[Warn] Synset %s not in synset.txt — skipping\n",synset.c_str());
            continue;
        }

        DIR* sub=opendir(cls_path.c_str());
        if(!sub) continue;
        struct dirent* img_entry;
        while((img_entry=readdir(sub))!=nullptr){
            if(img_entry->d_type==DT_REG||img_entry->d_type==DT_UNKNOWN){
                string n=img_entry->d_name;
                if(n.size()<4) continue;
                string ext=n.substr(n.find_last_of('.')+1);
                transform(ext.begin(),ext.end(),ext.begin(),::tolower);
                if(ext=="jpg"||ext=="jpeg"||ext=="png"||ext=="jpeg"){
                    ImageEntry e;
                    e.path         = cls_path+"/"+n;
                    e.name         = synset+"/"+n;
                    e.ground_truth = gt_class;
                    entries.push_back(e);
                }
            }
        }
        closedir(sub);
    }
    closedir(top);
    sort(entries.begin(),entries.end(),
         [](const ImageEntry& a,const ImageEntry& b){ return a.name<b.name; });
}

static void LoadWords(const string& path, vector<string>& kinds){
    kinds.clear(); ifstream f(path);
    if(!f){fprintf(stderr,"[Error] can't open %s\n",path.c_str()); exit(1);}
    string line; while(getline(f,line)) kinds.push_back(line);
}

// -----------------------------------------------------------------------------
// SBU/MBU FLIP PRIMITIVE
// -----------------------------------------------------------------------------
struct FlipInfo { size_t offset; int bit; uint8_t before; uint8_t after; };

static vector<FlipInfo> inject_sbu(uint8_t* base, size_t sz,
                                    int k, mt19937& rng, bool verbose,
                                    const char* tag){
    vector<FlipInfo> flips;
    if(!base||sz==0||k<=0) return flips;
    uniform_int_distribution<size_t> bdist(0,sz-1);
    uniform_int_distribution<int>    bitdist(0,7);
    set<size_t> used; int tries=0;
    while((int)flips.size()<k && tries<k*20){
        size_t off=bdist(rng);
        if(used.count(off)){tries++;continue;}
        used.insert(off);
        int bit=bitdist(rng);
        uint8_t orig=base[off];
        base[off]^=(uint8_t)(1u<<bit);
        flips.push_back({off,bit,orig,base[off]});
        if(verbose)
            sim_log("  [MBU][%s] offset=%7zu bit%d 0x%02X->0x%02X\n",
                    tag,off,bit,orig,base[off]);
        tries++;
    }
    return flips;
}

static void restore_flips(uint8_t* base, const vector<FlipInfo>& flips){
    for(auto& f:flips) base[f.offset]=f.before;
}

// -----------------------------------------------------------------------------
// FIND BYTES IN BINARY  (for locating mc_code / weights in xmodel file)
// -----------------------------------------------------------------------------
static size_t find_bytes_offset(const vector<uint8_t>& haystack,
                                 const vector<char>& needle){
    if(needle.empty()||needle.size()>haystack.size()) return string::npos;
    size_t sig=min((size_t)32,needle.size());
    for(size_t i=0;i+sig<=haystack.size();i++){
        if(memcmp(haystack.data()+i,needle.data(),sig)==0){
            size_t vlen=min((size_t)256,needle.size());
            if(memcmp(haystack.data()+i,needle.data(),vlen)==0)
                return i;
        }
    }
    return string::npos;
}

// -----------------------------------------------------------------------------
// CACHE REGION OFFSETS (once at startup)
// -----------------------------------------------------------------------------
static void cache_region_offsets(const xir::Subgraph* sg,
                                  const vector<uint8_t>& clean_model){
    auto* sgm=const_cast<xir::Subgraph*>(sg);
    try{
        auto mc=sgm->get_attr<vector<char>>("mc_code");
        if(!mc.empty()){
            g_mc_code_size  =mc.size();
            g_mc_code_offset=find_bytes_offset(clean_model,mc);
            sim_log("[Cache] mc_code: file_offset=%zu  size=%zu bytes\n",
                    g_mc_code_offset,g_mc_code_size);
        }
    }catch(...){sim_log("[Cache] mc_code attr not available\n");}
    try{
        auto wmap=sgm->get_attr<map<string,vector<char>>>("reg_id_to_parameter_value");
        for(auto& [rid,d]:wmap){
            if(d.empty()) continue;
            g_weight_size  =d.size();
            g_weight_offset=find_bytes_offset(clean_model,d);
            sim_log("[Cache] weights[%s]: file_offset=%zu  size=%zu bytes\n",
                    rid.c_str(),g_weight_offset,g_weight_size);
            break;
        }
    }catch(...){sim_log("[Cache] weight attr not available\n");}
}

// -----------------------------------------------------------------------------
// INFERENCE HELPER
// -----------------------------------------------------------------------------
struct InferenceResult {
    bool    ok=false,timed_out=false,exception=false,output_bad=false;
    int     top1=-1;
    float   top1_prob=0.f;
    int     top_k[TOP_K]={};
    float   top_k_prob[TOP_K]={};
};

static InferenceResult run_inference(vart::Runner* runner,
                                      int8_t* imgBuf, int inSz,int inH,int inW,
                                      int8_t* fcBuf,  int outSz, float out_scale,
                                      const xir::Tensor* inT,
                                      const xir::Tensor* outT){
    InferenceResult R;
    auto idims=inT->get_shape();  idims[0]=1;
    auto odims=outT->get_shape(); odims[0]=1;
    vector<unique_ptr<vart::TensorBuffer>> ib,ob;
    vector<shared_ptr<xir::Tensor>> bt;
    bt.push_back(shared_ptr<xir::Tensor>(
        xir::Tensor::create(inT->get_name(),idims,xir::DataType{xir::DataType::XINT,8u})));
    ib.push_back(make_unique<CpuFlatTensorBuffer>(imgBuf,bt.back().get()));
    bt.push_back(shared_ptr<xir::Tensor>(
        xir::Tensor::create(outT->get_name(),odims,xir::DataType{xir::DataType::XINT,8u})));
    ob.push_back(make_unique<CpuFlatTensorBuffer>(fcBuf,bt.back().get()));
    vector<vart::TensorBuffer*> ip={ib[0].get()},op={ob[0].get()};
    int ret=run_with_timeout(runner,ip,op,INFERENCE_TIMEOUT_MS);
    if(ret==1){R.timed_out=true; return R;}
    if(ret==2){R.exception=true; return R;}
    if(!output_tensor_sane(fcBuf,outSz)){R.output_bad=true;}
    vector<float> sm(outSz);
    CPUCalcSoftmax(fcBuf,outSz,sm.data(),out_scale);
    if(softmax_anomalous(sm.data(),outSz)){R.output_bad=true;}
    auto tk=topk(sm.data(),outSz,TOP_K);
    R.top1=tk[0]; R.top1_prob=sm[tk[0]];
    for(int i=0;i<TOP_K;i++){R.top_k[i]=tk[i]; R.top_k_prob[i]=sm[tk[i]];}
    R.ok=true; return R;
}

// -----------------------------------------------------------------------------
// BASELINE — run clean model on one image; ground truth comes from folder
// -----------------------------------------------------------------------------
static BaselineResult compute_baseline(vart::Runner* runner,
                                        const ImageEntry& entry,
                                        const vector<string>& kinds){
    BaselineResult B;
    B.image_name         = entry.name;
    B.image_path         = entry.path;
    B.ground_truth_class = entry.ground_truth;
    B.ground_truth_name  = (entry.ground_truth>=0&&entry.ground_truth<(int)kinds.size())
                            ? kinds[entry.ground_truth] : "?";

    auto outT=runner->get_output_tensors();
    auto inT =runner->get_input_tensors();
    float in_sc =get_input_scale(inT[0]);
    float out_sc=get_output_scale(outT[0]);
    int outSz=shapes.outTensorList[0].size;
    int inSz =shapes.inTensorList[0].size;
    int inH  =shapes.inTensorList[0].height;
    int inW  =shapes.inTensorList[0].width;

    vector<int8_t> imgBuf(inSz,0), fcBuf(outSz,0);
    Mat raw=imread(entry.path);
    if(raw.empty()){ sim_log("[Baseline] Cannot read %s\n",entry.path.c_str()); return B; }
    preprocess_image(raw,imgBuf.data(),inH,inW,in_sc);
    auto IR=run_inference(runner,imgBuf.data(),inSz,inH,inW,
                          fcBuf.data(),outSz,out_sc,inT[0],outT[0]);
    if(!IR.ok){ sim_log("[Baseline] Inference failed: %s\n",B.image_name.c_str()); return B; }

    B.baseline_class = IR.top1;
    B.baseline_prob  = IR.top1_prob;
    B.baseline_name  = (IR.top1>=0&&IR.top1<(int)kinds.size())?kinds[IR.top1]:"?";
    B.valid = true;
    return B;
}

// -----------------------------------------------------------------------------
// SUBPROCESS — isolated child process for INSTRUCTIONS/WEIGHTS injection
// Pipe: "OK c0 p0 c1 p1 c2 p2\n"
// -----------------------------------------------------------------------------
static bool run_in_subprocess(
    const string& corrupted_path,
    const int8_t* imgBuf,
    int outSz, float out_scale,
    bool verbose,
    vart::Runner*& runner,
    const xir::Subgraph* sg,
    const BaselineResult& B,
    RunResultMBU& RES)
{
    int pfd[2];
    if(pipe(pfd)!=0){RES.crash=true;return false;}
    pid_t pid=fork();
    if(pid<0){close(pfd[0]);close(pfd[1]);RES.crash=true;return false;}

    if(pid==0){
        // CHILD: close all inherited XRT/DRM fds to free DDR4 allocations
        int maxfd = getdtablesize();
        for(int fd=3; fd<maxfd; fd++) if(fd!=pfd[1]) close(fd);
        auto die=[&](){write(pfd[1],"DUE\n",4);close(pfd[1]);_exit(1);};
        try{
            auto cg  =xir::Graph::deserialize(corrupted_path);
            auto csgv=get_dpu_subgraph(cg.get());
            if(csgv.empty()) die();
            auto cr=vart::Runner::create_runner(csgv[0],"run");
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
            auto fut=cr->execute_async(ip,op);
            cr->wait(fut.first,-1);
            vector<float> sm(outSz,0.f),tmp(coutSz);
            CPUCalcSoftmax(cfcBuf.data(),coutSz,tmp.data(),out_scale);
            int n=min(coutSz,outSz); for(int i=0;i<n;i++) sm[i]=tmp[i];
            auto tk=topk(sm.data(),outSz,3);
            char buf[256];
            int len=snprintf(buf,sizeof(buf),"OK %d %.8f %d %.8f %d %.8f\n",
                tk[0],sm[tk[0]],tk[1],sm[tk[1]],tk[2],sm[tk[2]]);
            write(pfd[1],buf,len);
            close(pfd[1]); _exit(0);
        }catch(...){die();}
    }

    close(pfd[1]);
    const int WAIT_MS=INFERENCE_TIMEOUT_MS+15000;
    auto t0c=steady_clock::now(); int wst=0;
    while(true){
        pid_t wr=waitpid(pid,&wst,WNOHANG);
        if(wr==pid) break;
        if(duration_cast<milliseconds>(steady_clock::now()-t0c).count()>WAIT_MS){
            kill(pid,SIGKILL); waitpid(pid,&wst,0); break;
        }
        this_thread::sleep_for(milliseconds(100));
    }
    char rbuf[256]={}; read(pfd[0],rbuf,sizeof(rbuf)-1);
    close(pfd[0]);
    unlink(corrupted_path.c_str());

    bool ok=(WIFEXITED(wst)&&WEXITSTATUS(wst)==0);
    if(!ok){
        sim_log("[%s] Subprocess died DUE. Resetting DPU...\n",B.image_name.c_str());
        hardware_reset_dpu();
        this_thread::sleep_for(milliseconds(2000));
        auto nr=recreate_runner(sg);
        if(nr){runner=nr.release();}
        RES.timeout=true; return true;
    }
    if(strncmp(rbuf,"OK ",3)!=0){RES.crash=true; return true;}
    int c0,c1,c2; float p0,p1,p2;
    if(sscanf(rbuf+3,"%d %f %d %f %d %f",&c0,&p0,&c1,&p1,&c2,&p2)<6){
        RES.crash=true; return true;
    }
    RES.faulty_class[0]=c0; RES.faulty_prob[0]=p0;
    RES.faulty_class[1]=c1; RES.faulty_prob[1]=p1;
    RES.faulty_class[2]=c2; RES.faulty_prob[2]=p2;
    RES.correctly_classified = (c0==B.ground_truth_class);
    RES.prob_drop            = B.baseline_prob-p0;
    if(verbose)
        sim_log("[%s] k=%d gt=%d base=%d(%.3f) faulty=%d(%.3f) %s\n",
                B.image_name.c_str(),RES.k_bits,
                B.ground_truth_class,B.baseline_class,B.baseline_prob,
                c0,p0,RES.correctly_classified?"CORRECT":"WRONG");
    return true;
}

// -----------------------------------------------------------------------------
// SINGLE FAULTY RUN — one image, one fault injection
// -----------------------------------------------------------------------------
static bool perform_faulty_run(
    vart::Runner*& runner,
    const xir::Subgraph* sg,
    const vector<uint8_t>& clean_model,
    const vector<int8_t>& imgBuf,
    const BaselineResult& B,
    const vector<string>& kinds,
    FaultTarget target, int k, bool verbose,
    int run_id, mt19937& rng, RunResultMBU& RES)
{
    FaultTarget eff=target;
    if(eff==FaultTarget::ALL){
        static const FaultTarget pool[]={
            FaultTarget::INSTRUCTIONS,FaultTarget::WEIGHTS,
            FaultTarget::FEATURE_MAPS,FaultTarget::BUFFERS};
        eff=pool[rng()%4];
    }
    RES.k_bits             =k;
    RES.target_used        =eff;
    RES.image_name         =B.image_name;
    RES.ground_truth_class =B.ground_truth_class;
    RES.ground_truth_name  =B.ground_truth_name;
    RES.baseline_class     =B.baseline_class;
    RES.baseline_name      =B.baseline_name;
    RES.baseline_prob      =B.baseline_prob;

    auto outT=runner->get_output_tensors();
    auto inT =runner->get_input_tensors();
    float out_scale=get_output_scale(outT[0]);
    int outSz=shapes.outTensorList[0].size;
    int inSz =shapes.inTensorList[0].size;
    int inH  =shapes.inTensorList[0].height;
    int inW  =shapes.inTensorList[0].width;

    vector<int8_t> img(imgBuf);
    vector<int8_t> fcBuf(outSz,0);

    if(eff==FaultTarget::INSTRUCTIONS||eff==FaultTarget::WEIGHTS){
        if(clean_model.empty()){RES.crash=true;return false;}
        size_t attr_off=(eff==FaultTarget::INSTRUCTIONS)?g_mc_code_offset:g_weight_offset;
        size_t attr_sz =(eff==FaultTarget::INSTRUCTIONS)?g_mc_code_size  :g_weight_size;
        if(attr_off==string::npos||attr_sz==0){RES.crash=true;return false;}
        vector<uint8_t> patched=clean_model;
        auto pf=inject_sbu(patched.data()+attr_off,attr_sz,k,rng,verbose,targetName(eff).c_str());
        if(!pf.empty()){RES.fault_byte_offset=attr_off+pf[0].offset;RES.fault_bit=pf[0].bit;}
        {ofstream out(CORRUPTED_MODEL_PATH,ios::binary);
         if(!out){RES.crash=true;return false;}
         out.write(reinterpret_cast<const char*>(patched.data()),patched.size());}
        return run_in_subprocess(CORRUPTED_MODEL_PATH,img.data(),outSz,out_scale,
                                  verbose,runner,sg,B,RES);
    }

    if(eff==FaultTarget::FEATURE_MAPS){
        auto f=inject_sbu(reinterpret_cast<uint8_t*>(img.data()),
                          (size_t)inSz,k,rng,verbose,"feature_maps");
        if(!f.empty()){RES.fault_byte_offset=f[0].offset;RES.fault_bit=f[0].bit;}
    }

    auto IR=run_inference(runner,img.data(),inSz,inH,inW,
                          fcBuf.data(),outSz,out_scale,inT[0],outT[0]);
    if(IR.timed_out){
        RES.timeout=true;
        auto nr=recreate_runner(sg);
        if(nr){runner=nr.release();RES.recovered=true;}
        return true;
    }
    if(IR.exception){RES.crash=true;return false;}

    if(eff==FaultTarget::BUFFERS){
        auto f=inject_sbu(reinterpret_cast<uint8_t*>(fcBuf.data()),
                          (size_t)outSz,k,rng,verbose,"buffers");
        if(!f.empty()){RES.fault_byte_offset=f[0].offset;RES.fault_bit=f[0].bit;}
        if(!output_tensor_sane(fcBuf.data(),outSz)){RES.output_anomaly=true;}
        vector<float> sm(outSz);
        CPUCalcSoftmax(fcBuf.data(),outSz,sm.data(),out_scale);
        if(softmax_anomalous(sm.data(),outSz)){RES.output_anomaly=true;}
        auto tk=topk(sm.data(),outSz,3);
        IR.top1=tk[0]; IR.top1_prob=sm[tk[0]];
        for(int i=0;i<3;i++){IR.top_k[i]=tk[i];IR.top_k_prob[i]=sm[tk[i]];}
    }
    if(IR.output_bad) RES.output_anomaly=true;

    for(int i=0;i<3;i++){
        RES.faulty_class[i]=IR.top_k[i];
        RES.faulty_prob[i] =IR.top_k_prob[i];
        RES.faulty_name[i] =(IR.top_k[i]>=0&&IR.top_k[i]<(int)kinds.size())
                             ?kinds[IR.top_k[i]]:"?";
    }
    RES.correctly_classified=(IR.top1==B.ground_truth_class);
    RES.prob_drop           =B.baseline_prob-IR.top1_prob;

    if(verbose)
        sim_log("[%s] k=%d gt=%d base=%d(%.3f) faulty=%d(%.3f) %s\n",
                B.image_name.c_str(),k,
                B.ground_truth_class,B.baseline_class,B.baseline_prob,
                IR.top1,IR.top1_prob,RES.correctly_classified?"CORRECT":"WRONG");
    return true;
}

static void write_per_bit_csv(const vector<RunResultMBU>& results,
                               int k, const string& outDir){
    string path=outDir+"/results_k"+to_string(k)+"_bits.csv";
    ofstream f(path);
    if(!f){fprintf(stderr,"[CSV] Cannot write %s\n",path.c_str());return;}

    f<<"image_name,"
      "ground_truth_class,ground_truth_name,"
      "baseline_class,baseline_name,baseline_prob,"
      "faulty_top1,faulty_top1_name,faulty_top1_prob,"
      "faulty_top2,faulty_top2_name,faulty_top2_prob,"
      "faulty_top3,faulty_top3_name,faulty_top3_prob,"
      "correctly_classified,prob_drop,timeout,crash\n";

    for(auto& R:results){
        auto q=[](const string& s)->string{ return "\""+s+"\""; };
        f<<q(R.image_name)<<","
         <<R.ground_truth_class<<","<<q(R.ground_truth_name)<<","
         <<R.baseline_class<<","<<q(R.baseline_name)<<","
         <<fixed<<setprecision(6)<<R.baseline_prob<<","
         <<R.faulty_class[0]<<","<<q(R.faulty_name[0])<<","<<R.faulty_prob[0]<<","
         <<R.faulty_class[1]<<","<<q(R.faulty_name[1])<<","<<R.faulty_prob[1]<<","
         <<R.faulty_class[2]<<","<<q(R.faulty_name[2])<<","<<R.faulty_prob[2]<<","
         <<(R.correctly_classified?1:0)<<","
         <<R.prob_drop<<","
         <<(R.timeout?1:0)<<","
         <<(R.crash?1:0)<<"\n";
    }
    printf("[CSV] Saved: %s\n",path.c_str());
}

// -----------------------------------------------------------------------------
// CSV — accuracy summary across all bit counts
// columns: bits, total_images, baseline_correctly_classified, baseline_accuracy_pct,
//          correctly_classified, misclassified, accuracy_pct
// -----------------------------------------------------------------------------
struct AccuracyRow {
    int   bits;
    int   total_images;
    int   baseline_correct;
    float baseline_accuracy_pct;
    int   correctly_classified;
    int   misclassified;
    float accuracy_pct;
};

static void write_accuracy_csv(const vector<AccuracyRow>& rows,
                                const string& outDir){
    string path=outDir+"/accuracy_summary.csv";
    ofstream f(path);
    if(!f){fprintf(stderr,"[CSV] Cannot write %s\n",path.c_str());return;}
    f<<"bits,total_images,"
      "baseline_correctly_classified,baseline_accuracy_pct,"
      "correctly_classified,misclassified,accuracy_pct\n";
    for(auto& r:rows){
        f<<r.bits<<","<<r.total_images<<","
         <<r.baseline_correct<<","
         <<fixed<<setprecision(2)<<r.baseline_accuracy_pct<<","
         <<r.correctly_classified<<","<<r.misclassified<<","
         <<r.accuracy_pct<<"\n";
    }
    printf("[CSV] Saved: %s\n",path.c_str());
}

// -----------------------------------------------------------------------------
// PYTHON PLOT SCRIPT — generated and saved to output dir
// -----------------------------------------------------------------------------
static void write_plot_script(const string& outDir, const vector<int>& bit_counts){
    string path=outDir+"/plot_results.py";
    FILE* f=fopen(path.c_str(),"w");
    if(!f){fprintf(stderr,"[Plot] Cannot write %s\n",path.c_str());return;}

    fprintf(f,"#!/usr/bin/env python3\n");
    fprintf(f,"# MBU Simulation Result Plots\n");
    fprintf(f,"# Generated by MBU_simulate.cc\n");
    fprintf(f,"# Run on host: python3 plot_results.py\n\n");
    fprintf(f,"import os, sys\n");
    fprintf(f,"import pandas as pd\n");
    fprintf(f,"import matplotlib.pyplot as plt\n");
    fprintf(f,"import matplotlib.ticker as mticker\n");
    fprintf(f,"import numpy as np\n\n");
    fprintf(f,"OUTDIR = os.path.dirname(os.path.abspath(__file__))\n\n");

    // Plot 1: accuracy vs bits with baseline reference line
    fprintf(f,"# --- Plot 1: Accuracy vs Bits (with baseline) ---\n");
    fprintf(f,"acc_path = os.path.join(OUTDIR, 'accuracy_summary.csv')\n");
    fprintf(f,"if os.path.exists(acc_path):\n");
    fprintf(f,"    df_acc = pd.read_csv(acc_path)\n");
    fprintf(f,"    baseline_acc = df_acc['baseline_accuracy_pct'].iloc[0]\n");
    fprintf(f,"    # Build x-axis: prepend k=0 (baseline) then fault bits\n");
    fprintf(f,"    x_labels = ['0 (baseline)'] + df_acc['bits'].astype(str).tolist()\n");
    fprintf(f,"    y_vals   = [baseline_acc]   + df_acc['accuracy_pct'].tolist()\n");
    fprintf(f,"    colors   = ['forestgreen'] + ['steelblue']*len(df_acc)\n");
    fprintf(f,"    fig, ax = plt.subplots(figsize=(10, 5))\n");
    fprintf(f,"    bars = ax.bar(x_labels, y_vals, color=colors, edgecolor='black', width=0.5)\n");
    fprintf(f,"    ax.axhline(baseline_acc, color='forestgreen', linestyle='--',\n");
    fprintf(f,"               linewidth=1.2, label=f'Baseline {baseline_acc:.1f}%%')\n");
    fprintf(f,"    ax.set_xlabel('Number of Flipped Bits (k)', fontsize=12)\n");
    fprintf(f,"    ax.set_ylabel('Accuracy (ground truth)', fontsize=12)\n");
    fprintf(f,"    ax.set_title('MBU Fault Injection: Accuracy vs Bit Count', fontsize=13)\n");
    fprintf(f,"    ax.set_ylim(0, 105)\n");
    fprintf(f,"    ax.yaxis.set_major_formatter(mticker.PercentFormatter())\n");
    fprintf(f,"    ax.legend(fontsize=10)\n");
    fprintf(f,"    for i, v in enumerate(y_vals):\n");
    fprintf(f,"        ax.text(i, v+1.5, f'{v:.1f}%%', ha='center', fontsize=9)\n");
    fprintf(f,"    plt.tight_layout()\n");
    fprintf(f,"    plt.savefig(os.path.join(OUTDIR, 'plot_accuracy_vs_bits.png'), dpi=150)\n");
    fprintf(f,"    plt.close()\n");
    fprintf(f,"    print('[Plot] Saved: plot_accuracy_vs_bits.png')\n");
    fprintf(f,"else:\n");
    fprintf(f,"    print('[Warn] Not found:', acc_path)\n\n");

    // bit_counts list
    fprintf(f,"bit_counts = [");
    for(size_t i=0;i<bit_counts.size();i++)
        fprintf(f,"%d%s",bit_counts[i],i+1<bit_counts.size()?",":"");
    fprintf(f,"]\n\n");

    // Plot 2: prob_drop per image per bit count
    fprintf(f,"# --- Plot 2: Prob Drop per Image (one plot per bit count) ---\n");
    fprintf(f,"for k in bit_counts:\n");
    fprintf(f,"    csv_path = os.path.join(OUTDIR, f'results_k{k}_bits.csv')\n");
    fprintf(f,"    if not os.path.exists(csv_path):\n");
    fprintf(f,"        print('[Warn] Not found:', csv_path)\n");
    fprintf(f,"        continue\n");
    fprintf(f,"    df = pd.read_csv(csv_path)\n");
    fprintf(f,"    df_valid = df[(df['timeout']==0) & (df['crash']==0)].copy()\n");
    fprintf(f,"    if df_valid.empty:\n");
    fprintf(f,"        print(f'[Warn] No valid runs for k={k}')\n");
    fprintf(f,"        continue\n");
    fprintf(f,"    avg_drop = df_valid.groupby('image_name')['prob_drop'].mean().reset_index()\n");
    fprintf(f,"    avg_drop = avg_drop.sort_values('image_name')\n");
    fprintf(f,"    short_names = [os.path.splitext(n)[0][-20:] for n in avg_drop['image_name']]\n");
    fprintf(f,"    fig, ax = plt.subplots(figsize=(max(8, len(short_names)*0.8), 5))\n");
    fprintf(f,"    colors = ['tomato' if v > 0.05 else 'steelblue' for v in avg_drop['prob_drop']]\n");
    fprintf(f,"    ax.bar(short_names, avg_drop['prob_drop'], color=colors, edgecolor='black')\n");
    fprintf(f,"    ax.axhline(0, color='black', linewidth=0.8, linestyle='--')\n");
    fprintf(f,"    ax.set_xlabel('Image', fontsize=11)\n");
    fprintf(f,"    ax.set_ylabel('Avg Prob Drop (golden - faulty)', fontsize=10)\n");
    fprintf(f,"    ax.set_title(f'MBU k={k} bits: Probability Drop per Image', fontsize=12)\n");
    fprintf(f,"    plt.xticks(rotation=45, ha='right', fontsize=8)\n");
    fprintf(f,"    plt.tight_layout()\n");
    fprintf(f,"    out_path = os.path.join(OUTDIR, f'plot_prob_drop_k{k}.png')\n");
    fprintf(f,"    plt.savefig(out_path, dpi=150)\n");
    fprintf(f,"    plt.close()\n");
    fprintf(f,"    print(f'[Plot] Saved: plot_prob_drop_k{k}.png')\n\n");

    fprintf(f,"print('\\n[Done] All plots saved to:', OUTDIR)\n");
    fprintf(f,"for fn in sorted(os.listdir(OUTDIR)):\n");
    fprintf(f,"    if fn.endswith('.png'):\n");
    fprintf(f,"        print('  ', fn)\n");

    fclose(f);
    printf("[Script] Plot script saved: %s\n",path.c_str());
    printf("[Script] Run on host: python3 %s\n",path.c_str());
}

// -----------------------------------------------------------------------------
// PARSE TARGET
// -----------------------------------------------------------------------------
static FaultTarget parse_target(const string& s){
    string lo=s; transform(lo.begin(),lo.end(),lo.begin(),::tolower);
    if(lo=="instructions") return FaultTarget::INSTRUCTIONS;
    if(lo=="weights")      return FaultTarget::WEIGHTS;
    if(lo=="feature_maps"||lo=="featuremaps"||lo=="input")
                           return FaultTarget::FEATURE_MAPS;
    if(lo=="buffers"||lo=="output") return FaultTarget::BUFFERS;
    if(lo=="all")          return FaultTarget::ALL;
    fprintf(stderr,"[Config] Unknown target '%s', using feature_maps\n",s.c_str());
    return FaultTarget::FEATURE_MAPS;
}

// -----------------------------------------------------------------------------
// MAIN
// -----------------------------------------------------------------------------
int main(int argc, char* argv[]){
    if(argc<2){
        printf("Usage: %s <model.xmodel> [target] [-v]\n",argv[0]);
        printf("  val folder assumed at ./val/  (ImageNet-mini structure)\n");
        printf("  target: instructions|weights|feature_maps|buffers|all\n");
        return -1;
    }

    install_crash_handlers();
    mt19937 rng(static_cast<uint32_t>(time(nullptr))^(uint32_t)getpid());

    SimConfig cfg;
    cfg.model_path = argv[1];
    if(argc>=3) cfg.target=parse_target(argv[2]);
    cfg.verbose=(argc>=4&&string(argv[3])=="-v");

    // -- Interactive prompts -----------------------------------------------
    printf("\n--------------------------------------------\n");
    printf("   MBU Fault Injection Simulator — Setup\n");
    printf("--------------------------------------------\n\n");

    printf("Enter val folder path [default ./val_subset]: ");
    fflush(stdout);
    {
        string line; getline(cin,line);
        cfg.val_folder = line.empty() ? "./val_subset" : line;
    }

    printf("Enter bit counts (space-separated) [default 1 5 10 15 20]: ");
    fflush(stdout);
    {
        string line; getline(cin,line);
        istringstream iss(line); int v;
        while(iss>>v) if(v>0) cfg.bit_counts.push_back(v);
    }
    if(cfg.bit_counts.empty()){
        cfg.bit_counts={1,5,10,15,20};
    }
    sort(cfg.bit_counts.begin(),cfg.bit_counts.end());
    cfg.bit_counts.erase(unique(cfg.bit_counts.begin(),cfg.bit_counts.end()),
                          cfg.bit_counts.end());

    printf("Enter experiment name [default mbu_results]: ");
    fflush(stdout);
    {
        string line; getline(cin,line);
        if(!line.empty()) cfg.base_name=line;
    }
    // Root FaultResults dir created now; target subdirs created per-experiment below
    mkdirp("./FaultResults/"+cfg.base_name);

    if(argc<3){
        printf("Target [feature_maps/weights/instructions/buffers/all, default feature_maps]: ");
        fflush(stdout);
        string line; getline(cin,line);
        if(!line.empty()) cfg.target=parse_target(line);
    }

    printf("\n[Config] model      = %s\n", cfg.model_path.c_str());
    printf("[Config] val_folder = %s\n",  cfg.val_folder.c_str());
    printf("[Config] bits       =");
    for(int k:cfg.bit_counts) printf(" %d",k);
    printf("\n[Config] target     = %s\n", targetName(cfg.target).c_str());
    printf("[Config] results in = ./FaultResults/%s/<target>/\n\n", cfg.base_name.c_str());

    // -- Open log in the base experiment folder ---------------------------
    string base_exp_dir = "./FaultResults/" + cfg.base_name;
    string logpath = base_exp_dir + "/mbu_sim.log";
    g_logfp=fopen(logpath.c_str(),"w");
    if(!g_logfp) fprintf(stderr,"[Warn] Cannot open log %s\n",logpath.c_str());

    // -- Load labels + synset mapping -------------------------------------
    vector<string> kinds;
    LoadWords(wordsPath+"words.txt",kinds);

    map<string,int> synset_to_idx = LoadSynsets(wordsPath+"synset.txt");
    if(synset_to_idx.empty()){
        fprintf(stderr,"[Error] synset.txt empty or missing\n"); return -1;
    }
    sim_log("[Setup] %zu classes  |  %zu synsets\n",kinds.size(),synset_to_idx.size());

    // -- Walk val/ to collect images with ground truth labels -------------
    vector<ImageEntry> entries;
    ListImagesWithGroundTruth(cfg.val_folder, synset_to_idx, entries);
    if(entries.empty()){
        fprintf(stderr,"[Error] No images found in %s\n",cfg.val_folder.c_str());
        return -1;
    }
    sim_log("[Setup] %zu images with ground truth labels\n",entries.size());
    printf("[Setup] %zu images across %zu classes\n",
           entries.size(), synset_to_idx.size());

    // -- Load model + runner -----------------------------------------------
    auto graph   =xir::Graph::deserialize(cfg.model_path);
    auto subgraph=get_dpu_subgraph(graph.get());
    CHECK_EQ(subgraph.size(),1u)<<"Expected one DPU subgraph";

    auto runner_owned=vart::Runner::create_runner(subgraph[0],"run");
    vart::Runner* runner=runner_owned.get();

    auto inT =runner->get_input_tensors();
    auto outT=runner->get_output_tensors();
    static TensorShape insh[8],outsh[8];
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

    // -- Load clean model binary ------------------------------------------
    vector<uint8_t> clean_model;
    {
        ifstream mf(cfg.model_path,ios::binary);
        if(!mf){fprintf(stderr,"[Error] Cannot open model\n");return -1;}
        clean_model.assign((istreambuf_iterator<char>(mf)),istreambuf_iterator<char>());
        sim_log("[Setup] Model in RAM: %zu bytes\n",clean_model.size());
    }
    cache_region_offsets(subgraph[0],clean_model);

    float in_sc=get_input_scale(inT[0]);
    int inSz=shapes.inTensorList[0].size;
    int inH =shapes.inTensorList[0].height;
    int inW =shapes.inTensorList[0].width;

    // -- Compute baseline for ALL images ONCE (before any fault injection) -
    // This is the clean-model pass. We cache every image's baseline result
    // so we don't re-run clean inference for each bit count.
    printf("[Baseline] Running clean model on %zu images...\n",entries.size());
    vector<BaselineResult> baselines;
    baselines.reserve(entries.size());
    vector<vector<int8_t>> imgBufs;   // pre-processed pixel buffers, reused across k
    imgBufs.reserve(entries.size());

    for(size_t i=0;i<entries.size();i++){
        printf("\r[Baseline] %zu / %zu  ",i+1,entries.size()); fflush(stdout);
        BaselineResult B=compute_baseline(runner,entries[i],kinds);
        if(!B.valid){
            fprintf(stderr,"\n[Skip] Baseline failed: %s\n",entries[i].name.c_str());
            // Push invalid entry so indices stay aligned
        }
        baselines.push_back(B);

        // Pre-process and cache image buffer
        vector<int8_t> buf(inSz,0);
        Mat raw=imread(entries[i].path);
        if(!raw.empty()) preprocess_image(raw,buf.data(),inH,inW,in_sc);
        imgBufs.push_back(move(buf));
    }
    printf("\r[Baseline] Done.                    \n");

    // Print baseline accuracy (clean model vs ground truth)
    int base_correct=0, base_total=0;
    {
        for(auto& B:baselines){
            if(!B.valid) continue;
            base_total++;
            if(B.baseline_class==B.ground_truth_class) base_correct++;
        }
        float base_pct = base_total>0?100.f*base_correct/base_total:0.f;
        printf("[Baseline] Clean model accuracy: %d/%d = %.2f%%\n",
               base_correct,base_total,base_pct);
        sim_log("[Baseline] Clean accuracy: %d/%d = %.2f%%\n",
                base_correct,base_total,base_pct);
    }

    // -- OUTER LOOP: bit counts --------------------------------------------
    vector<AccuracyRow> accuracy_rows;

    // Create (or clear) the target subfolder once for this experiment.
    // Structure: ./FaultResults/<base_name>/<target>/
    string target_dir = prepare_target_dir(cfg.base_name, cfg.target);

    for(int k : cfg.bit_counts){
        sim_log("\n---------------- k=%d bits ----------------\n",k);
        printf("\n[Run] k=%d bits  (%zu images, single pass)\n",
               k,entries.size());

        vector<RunResultMBU> results_this_k;
        results_this_k.reserve(entries.size());
        int total_correct=0, img_total=0;

        for(size_t img_idx=0;img_idx<entries.size();img_idx++){
            const BaselineResult& B=baselines[img_idx];
            if(!B.valid) continue;

            printf("\r  [%zu/%zu] %s  k=%d  ",
                   img_idx+1,entries.size(),B.image_name.c_str(),k);
            fflush(stdout);

            RunResultMBU R;
            bool ok=perform_faulty_run(
                runner,subgraph[0],clean_model,
                imgBufs[img_idx],B,kinds,
                cfg.target,k,cfg.verbose,
                (int)img_idx,rng,R);

            if(!ok&&!R.timeout){
                sim_log("[Main] Hard crash img=%s\n",B.image_name.c_str());
                auto nr=recreate_runner(subgraph[0]);
                if(nr){runner=nr.release();R.recovered=true;}
            }

            for(int i=0;i<3;i++)
                if(R.faulty_name[i].empty()&&R.faulty_class[i]>=0
                   &&R.faulty_class[i]<(int)kinds.size())
                    R.faulty_name[i]=kinds[R.faulty_class[i]];

            if(R.correctly_classified) total_correct++;
            img_total++;
            results_this_k.push_back(R);
        }
        printf("\r  Done %d images                         \n",img_total);

        // Save per-bit CSV into target subfolder
        write_per_bit_csv(results_this_k, k, target_dir);

        float acc_pct  = img_total>0?100.f*total_correct/img_total:0.f;
        float base_pct = base_total>0?100.f*base_correct/base_total:0.f;

        AccuracyRow row;
        row.bits                 = k;
        row.total_images         = img_total;
        row.baseline_correct     = base_correct;
        row.baseline_accuracy_pct= base_pct;
        row.correctly_classified = total_correct;
        row.misclassified        = img_total-total_correct;
        row.accuracy_pct         = acc_pct;
        accuracy_rows.push_back(row);

        sim_log("[Summary] k=%-3d  baseline=%d/%d(%.2f%%)  faulty=%d/%d(%.2f%%)\n",
                k,base_correct,base_total,base_pct,total_correct,img_total,acc_pct);
        printf("[Summary] k=%-3d  baseline=%.2f%%  faulty=%.2f%%\n",
               k,base_pct,acc_pct);
    }

    // Save accuracy summary + plot script into target subfolder
    write_accuracy_csv(accuracy_rows, target_dir);
    write_plot_script(target_dir, cfg.bit_counts);

    // -- Print final table -------------------------------------------------
    float base_pct_final = base_total>0?100.f*base_correct/base_total:0.f;
    printf("\n--------------------------------------------\n");
    printf("  ACCURACY SUMMARY (ground truth = val folder)\n");
    printf("--------------------------------------------\n");
    printf("  Baseline (k=0): %d/%d = %.2f%%\n",
           base_correct, base_total, base_pct_final);
    printf("  ------------------------------------------\n");
    printf("  %-6s  %-10s  %-8s  %s\n","Bits","Faulty","Total","Accuracy");
    printf("  ------------------------------------------\n");
    for(auto& r:accuracy_rows)
        printf("  %-6d  %-10d  %-8d  %.2f%%  (baseline=%.2f%%)\n",
               r.bits,r.correctly_classified,r.total_images,
               r.accuracy_pct,r.baseline_accuracy_pct);
    printf("--------------------------------------------\n");

    if(g_logfp) fclose(g_logfp);

    printf("\n[Done] Results in: %s/\n", target_dir.c_str());
    printf("  Per-bit CSVs : results_k<N>_bits.csv\n");
    printf("  Accuracy CSV : accuracy_summary.csv\n");
    printf("  Plot script  : plot_results.py\n");
    printf("\nFull structure:\n");
    printf("  ./FaultResults/%s/\n", cfg.base_name.c_str());
    printf("    %s/  <- this run\n", targetDirName(cfg.target).c_str());
    printf("\nOn host: python3 %s/plot_results.py\n", target_dir.c_str());
    return 0;
}

