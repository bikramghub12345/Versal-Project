/*
 * SEFI_transient.cc  —  Transient SEFI Simulation (Layer-by-Layer, Weights Only)
 * ================================================================================
 * Platform : Xilinx ZCU104  |  DPUCZDX8G  |  PyTorch ResNet50 (18-piece split)
 * Reference : Guertin S.M., NEPP DDR4 Radiation Eval FY24, JPL/Caltech 2025.
 *
 * WORKFLOW:
 *   1. Inject SEFI (row/col/block) randomly into the conceptual 25 MB weight space.
 *   2. For each of the 18 pieces in execution order:
 *        a. Create a fresh VART runner (VART DMAs this piece's weights to CMA).
 *        b. Apply flips for this piece to its DDR4 weight region.
 *        c. Execute this piece (DPU reads corrupted weights).
 *        d. Restore flips immediately after wait() (transient clears).
 *        e. Requantize output to match next piece's expected input scale.
 *        f. Copy output to CPU buffer; destroy runner (frees CMA).
 *   3. Final output → softmax → compare with baseline → record metrics.
 *
 * OUTPUT FILES (per mode run):
 *   results_<mode>.csv            — per-image: matches SEFI_simulate.cc format
 *   accuracy_summary.csv          — accuracy/precision/recall/F1 baseline vs faulty
 *   per_layer_details_<mode>.csv  — per-image per-piece: injection addresses, bytes, bits
 *   sefi_transient.log            — full verbose log (addresses, scales, injection details)
 *
 * BUILD:
 *   g++ -std=c++17 -O2 -o SEFI_transient src/SEFI_transient.cc \
 *       ../common/common.cpp -I ./src -I ../common \
 *       -I /usr/include/opencv4 -I /usr/include/vitis_ai \
 *       $(pkg-config --cflags --libs opencv4) \
 *       -lvart-runner -lxir -lglog -lpthread
 *
 * USAGE (root required): sudo ./SEFI_transient <models_dir> [row|column|block] [-v]
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
#include <cassert>
#include <cmath>
#include <cstdarg>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <map>
#include <memory>
#include <numeric>
#include <random>
#include <set>
#include <string>
#include <vector>

#include "common.h"
#include "piece_weight_sizes.h"
#include <opencv2/opencv.hpp>

using namespace std;
using namespace cv;

// =============================================================================
// CONSTANTS
// =============================================================================
#define TOP_K 5
static const size_t   DDR4_ROW_BYTES   = 8192;
static const size_t   DDR4_COL_DEFAULT = 8;
static const uint64_t DPU_CTRL_BASE    = 0x80000000ULL;
static const uint32_t OFF_BASE0_LO     = 0x60;
static const uint32_t OFF_BASE0_HI     = 0x64;

// =============================================================================
// SEFI MODE
// =============================================================================
enum class TransientMode { ROW, COLUMN, BLOCK };
static const char* mode_name(TransientMode m) {
    switch(m){ case TransientMode::ROW:    return "transient-SEFI-row";
               case TransientMode::COLUMN: return "transient-SEFI-col";
               case TransientMode::BLOCK:  return "transient-SEFI-blk"; }
    return "?";
}
static string mode_folder(TransientMode m) {
    switch(m){ case TransientMode::ROW:    return "02. transient-SEFI-row";
               case TransientMode::COLUMN: return "04. transient-SEFI-col";
               case TransientMode::BLOCK:  return "06. transient-SEFI-blk"; }
    return "00";
}

// =============================================================================
// GLOBALS
// =============================================================================
static int   g_devmem_fd = -1;
static FILE* g_logfp     = nullptr;

// =============================================================================
// UTILITY — identical to SEFI_simulate.cc
// =============================================================================
static void sim_log(const char* fmt, ...) {
    va_list a1,a2;
    va_start(a1,fmt); vprintf(fmt,a1); va_end(a1);
    if(g_logfp){va_start(a2,fmt);vfprintf(g_logfp,fmt,a2);va_end(a2);fflush(g_logfp);}
}
// Log only to file (not terminal)
static void log_only(const char* fmt, ...) {
    if(!g_logfp) return;
    va_list a; va_start(a,fmt); vfprintf(g_logfp,fmt,a); va_end(a); fflush(g_logfp);
}
static void mkdirp(const string& p) {
    string t=p;
    for(size_t i=1;i<t.size();i++){if(t[i]=='/'){t[i]='\0';mkdir(t.c_str(),0755);t[i]='/';}}
    mkdir(t.c_str(),0755);
}
static void clear_dir(const string& p) {
    DIR*d=opendir(p.c_str());if(!d)return;struct dirent*e;
    while((e=readdir(d))!=nullptr){string fn=e->d_name;if(fn=="."||fn=="..")continue;
    string fp=p+"/"+fn;struct stat s;lstat(fp.c_str(),&s);
    if(S_ISREG(s.st_mode))unlink(fp.c_str());}closedir(d);
}
static uint64_t read_ctrl_reg64(uint32_t lo, uint32_t hi) {
    if(g_devmem_fd<0)return 0;
    void*m=mmap(NULL,4096,PROT_READ,MAP_SHARED,g_devmem_fd,(off_t)(uint64_t)DPU_CTRL_BASE);
    if(m==MAP_FAILED)return 0;
    volatile uint32_t*r=(volatile uint32_t*)m;
    uint64_t v=((uint64_t)r[hi/4]<<32)|r[lo/4];munmap(m,4096);return v;
}
static uint8_t* region_map_rw(uint64_t phys,size_t sz,uint64_t&pg,size_t&adj,size_t&msz){
    pg=phys&~(uint64_t)4095;adj=(size_t)(phys-pg);msz=sz+adj;
    void*m=mmap(NULL,msz,PROT_READ|PROT_WRITE,MAP_SHARED,g_devmem_fd,(off_t)pg);
    if(m==MAP_FAILED){perror("[mmap_rw]");return nullptr;}
    return(uint8_t*)m+adj;
}
static void CPUCalcSoftmax(const int8_t*d,int sz,float*out,float scale){
    float mx=-1e30f;
    for(int i=0;i<sz;i++){float v=(float)d[i]*scale;if(v>mx)mx=v;}
    double sum=0;
    for(int i=0;i<sz;i++){out[i]=expf((float)d[i]*scale-mx);sum+=out[i];}
    for(int i=0;i<sz;i++)out[i]/=(float)sum;
}
static vector<int> topk(const float*p,int sz,int k){
    vector<int>idx(sz);iota(idx.begin(),idx.end(),0);
    partial_sort(idx.begin(),idx.begin()+k,idx.end(),[&](int a,int b){return p[a]>p[b];});
    idx.resize(k);return idx;
}

// =============================================================================
// IMAGE / LABEL LOADING — identical to SEFI_simulate.cc
// =============================================================================
struct ImageEntry { string path; string name; int ground_truth=-1; };

static void LoadWords(const string&p,vector<string>&k){
    k.clear();ifstream f(p);if(!f){fprintf(stderr,"Error:%s\n",p.c_str());exit(1);}
    string s;while(getline(f,s))k.push_back(s);
}
static map<string,int> LoadSynsets(const string&p){
    map<string,int>m;int i=0;ifstream f(p);
    if(!f){fprintf(stderr,"Error:%s\n",p.c_str());exit(1);}
    string s;while(getline(f,s)){m[s]=i++;}return m;
}
static void ListImagesWithGroundTruth(const string&val,
        const map<string,int>&syn,vector<ImageEntry>&out){
    out.clear();
    DIR*d=opendir(val.c_str());if(!d){fprintf(stderr,"No dir:%s\n",val.c_str());return;}
    struct dirent*e;
    while((e=readdir(d))!=nullptr){
        if(e->d_type!=DT_DIR)continue;
        string sn=e->d_name;if(sn=="."||sn=="..")continue;
        auto it=syn.find(sn);int gt=(it!=syn.end())?it->second:-1;
        string sub=val+"/"+sn;
        DIR*d2=opendir(sub.c_str());if(!d2)continue;struct dirent*e2;
        while((e2=readdir(d2))!=nullptr){
            if(e2->d_type!=DT_REG&&e2->d_type!=DT_UNKNOWN)continue;
            string fn=e2->d_name,fnl=fn;
            transform(fnl.begin(),fnl.end(),fnl.begin(),::tolower);
            bool ok=false;
            if(fnl.size()>4){auto e4=fnl.substr(fnl.size()-4);if(e4==".jpg"||e4==".png")ok=true;}
            if(!ok&&fnl.size()>5){if(fnl.substr(fnl.size()-5)==".jpeg")ok=true;}
            if(!ok)continue;
            ImageEntry ie;ie.path=sub+"/"+fn;ie.name=sn+"/"+fn;ie.ground_truth=gt;
            out.push_back(ie);
        }closedir(d2);
    }closedir(d);
    sort(out.begin(),out.end(),[](const ImageEntry&a,const ImageEntry&b){return a.name<b.name;});
}

// =============================================================================
// PYTORCH PREPROCESSING — matches resnet50_pt.xmodel quantisation
// =============================================================================
static void preprocess_image(const Mat&src,int8_t*dst,int inH,int inW,float scale){
    static const float mean[3]={0.485f,0.456f,0.406f};
    static const float std_[3]={0.229f,0.224f,0.225f};
    Mat rsz;resize(src,rsz,Size(inW,inH),0,0,INTER_LINEAR);
    for(int h=0;h<inH;h++) for(int w=0;w<inW;w++){
        Vec3b bgr=rsz.at<Vec3b>(h,w);
        float ch[3]={bgr[2]/255.f,bgr[1]/255.f,bgr[0]/255.f};
        for(int c=0;c<3;c++){
            float v=((ch[c]-mean[c])/std_[c])*scale;
            dst[h*inW*3+w*3+c]=(int8_t)max(-128.f,min(127.f,roundf(v)));
        }
    }
}

// =============================================================================
// PIECE INFO
// =============================================================================
struct PieceInfo {
    string name, xmodel_path;
    unique_ptr<xir::Graph> graph;
    const xir::Subgraph*   dpu_subgraph = nullptr;
    size_t   weight_size=0, conceptual_start=0;
    uint64_t weight_phys=0;
    string   in_tensor_name, out_tensor_name;
    vector<int> in_dims, out_dims;
    float    in_scale=1.f, out_scale=1.f;
    int      in_elems=0, out_elems=0;
    vector<int8_t> out_buf;
};

static void discover_piece(PieceInfo& p) {
    auto runner=vart::Runner::create_runner(p.dpu_subgraph,"run");
    auto inT=runner->get_input_tensors();
    auto outT=runner->get_output_tensors();
    p.in_tensor_name=inT[0]->get_name();
    p.out_tensor_name=outT[0]->get_name();
    p.in_dims=inT[0]->get_shape();
    p.out_dims=outT[0]->get_shape();
    p.in_scale=get_input_scale(inT[0]);
    p.out_scale=get_output_scale(outT[0]);
    p.in_elems=inT[0]->get_element_num();
    p.out_elems=outT[0]->get_element_num();
    p.out_buf.resize(p.out_elems,0);
    // dry run to populate DPU registers
    vector<int8_t>din(p.in_elems,0),dout(p.out_elems,0);
    auto di=p.in_dims;di[0]=1;auto dO=p.out_dims;dO[0]=1;
    vector<shared_ptr<xir::Tensor>>bt;
    bt.push_back(shared_ptr<xir::Tensor>(xir::Tensor::create(
        p.in_tensor_name,di,xir::DataType{xir::DataType::XINT,8u})));
    bt.push_back(shared_ptr<xir::Tensor>(xir::Tensor::create(
        p.out_tensor_name,dO,xir::DataType{xir::DataType::XINT,8u})));
    auto ib=make_unique<CpuFlatTensorBuffer>(din.data(),bt[0].get());
    auto ob=make_unique<CpuFlatTensorBuffer>(dout.data(),bt[1].get());
    vector<vart::TensorBuffer*>ip{ib.get()},op{ob.get()};
    auto j=runner->execute_async(ip,op);runner->wait(j.first,-1);
    p.weight_phys=read_ctrl_reg64(OFF_BASE0_LO,OFF_BASE0_HI);
}

// =============================================================================
// BYTE FLIP
// =============================================================================
struct ByteFlip {
    int piece_id; size_t local_offset; uint8_t xor_mask, original;
};

static void apply_flips(int kid,vector<ByteFlip>&flips,uint64_t wp,size_t ws){
    if(g_devmem_fd<0||wp==0||ws==0)return;
    bool any=false;for(auto&f:flips)if(f.piece_id==kid){any=true;break;}
    if(!any)return;
    uint64_t pg;size_t adj,msz;
    uint8_t*base=region_map_rw(wp,ws,pg,adj,msz);if(!base)return;
    for(auto&f:flips){
        if(f.piece_id!=kid||f.local_offset>=ws)continue;
        f.original=base[f.local_offset];base[f.local_offset]^=f.xor_mask;
    }
    munmap(base-adj,msz);
}
static void restore_flips(int kid,const vector<ByteFlip>&flips,uint64_t wp,size_t ws){
    if(g_devmem_fd<0||wp==0||ws==0)return;
    bool any=false;for(auto&f:flips)if(f.piece_id==kid){any=true;break;}
    if(!any)return;
    uint64_t pg;size_t adj,msz;
    uint8_t*base=region_map_rw(wp,ws,pg,adj,msz);if(!base)return;
    for(auto&f:flips){
        if(f.piece_id!=kid||f.local_offset>=ws)continue;
        base[f.local_offset]=f.original;
    }
    munmap(base-adj,msz);
}

// Execute one piece (ephemeral runner), optionally with flips
static bool exec_piece(PieceInfo&p,int kid,int8_t*in_buf,int8_t*out_buf,
                       vector<ByteFlip>*flips=nullptr){
    auto runner=vart::Runner::create_runner(p.dpu_subgraph,"run");
    if(flips)apply_flips(kid,*flips,p.weight_phys,p.weight_size);
    auto di=p.in_dims;di[0]=1;auto dO=p.out_dims;dO[0]=1;
    vector<shared_ptr<xir::Tensor>>bt;
    bt.push_back(shared_ptr<xir::Tensor>(xir::Tensor::create(
        p.in_tensor_name,di,xir::DataType{xir::DataType::XINT,8u})));
    bt.push_back(shared_ptr<xir::Tensor>(xir::Tensor::create(
        p.out_tensor_name,dO,xir::DataType{xir::DataType::XINT,8u})));
    auto ib=make_unique<CpuFlatTensorBuffer>(in_buf,bt[0].get());
    auto ob=make_unique<CpuFlatTensorBuffer>(out_buf,bt[1].get());
    vector<vart::TensorBuffer*>ip{ib.get()},op{ob.get()};
    bool ok=true;
    try{auto j=runner->execute_async(ip,op);runner->wait(j.first,-1);}catch(...){ok=false;}
    if(flips)restore_flips(kid,*flips,p.weight_phys,p.weight_size);
    return ok;
}

// Requantize output of piece k to match input scale of piece k+1
static void requantize(vector<PieceInfo>&pieces,int k){
    if(k>=N_PIECES-1)return;
    float ratio=pieces[k].out_scale*pieces[k+1].in_scale;
    if(fabs(ratio-1.f)<0.001f)return;
    for(int j=0;j<pieces[k].out_elems;j++){
        float v=(float)pieces[k].out_buf[j]*ratio;
        pieces[k].out_buf[j]=(int8_t)max(-128.f,min(127.f,roundf(v)));
    }
}

// =============================================================================
// INJECTION PLANNING
// =============================================================================
static vector<ByteFlip> plan_row(mt19937&rng,const vector<PieceInfo>&p,size_t total){
    vector<ByteFlip>f;if(total<DDR4_ROW_BYTES)return f;
    size_t rs=uniform_int_distribution<size_t>(0,total/DDR4_ROW_BYTES-1)(rng)*DDR4_ROW_BYTES;
    size_t re=rs+DDR4_ROW_BYTES;
    uniform_int_distribution<uint8_t>md(1,255);
    for(int k=0;k<(int)p.size();k++){
        size_t ps=p[k].conceptual_start,pe=ps+p[k].weight_size;
        size_t os=max(rs,ps),oe=min(re,pe);if(os>=oe)continue;
        for(size_t off=os;off<oe;off++){
            ByteFlip bf;bf.piece_id=k;bf.local_offset=off-ps;
            bf.xor_mask=md(rng);bf.original=0;f.push_back(bf);
        }
    }
    log_only("[row] conceptual_start=%zu end=%zu flips=%zu\n",rs,re,f.size());
    return f;
}
static vector<ByteFlip> plan_column(mt19937&rng,size_t cw,const vector<PieceInfo>&p){
    vector<ByteFlip>f;if(cw==0||cw>DDR4_ROW_BYTES)cw=DDR4_COL_DEFAULT;
    size_t cs=uniform_int_distribution<size_t>(0,DDR4_ROW_BYTES-cw)(rng);
    uniform_int_distribution<uint8_t>md(1,255);
    for(int k=0;k<(int)p.size();k++){
        if(p[k].weight_size==0)continue;
        size_t nc=p[k].weight_size/DDR4_ROW_BYTES,rem=p[k].weight_size%DDR4_ROW_BYTES;
        for(size_t row=0;row<nc+(rem>0?1:0);row++){
            size_t ra=(row<nc)?DDR4_ROW_BYTES:rem;
            if(cs>=ra)continue;
            size_t bw=min(cw,ra-cs);
            for(size_t c=0;c<bw;c++){
                ByteFlip bf;bf.piece_id=k;bf.local_offset=row*DDR4_ROW_BYTES+cs+c;
                bf.xor_mask=md(rng);bf.original=0;f.push_back(bf);
            }
        }
    }
    log_only("[col] col_start=%zu col_width=%zu flips=%zu\n",cs,cw,f.size());
    return f;
}
static vector<ByteFlip> plan_block(mt19937&rng,size_t bsz,
        const vector<PieceInfo>&p,size_t total){
    vector<ByteFlip>f;if(total==0||bsz==0)return f;
    bsz=min(bsz,total);
    size_t bs=uniform_int_distribution<size_t>(0,total-bsz)(rng),be=bs+bsz;
    uniform_int_distribution<uint8_t>md(1,255);
    for(int k=0;k<(int)p.size();k++){
        size_t ps=p[k].conceptual_start,pe=ps+p[k].weight_size;
        size_t os=max(bs,ps),oe=min(be,pe);if(os>=oe)continue;
        for(size_t off=os;off<oe;off++){
            ByteFlip bf;bf.piece_id=k;bf.local_offset=off-ps;
            bf.xor_mask=md(rng);bf.original=0;f.push_back(bf);
        }
    }
    log_only("[block] start=%zu end=%zu flips=%zu\n",bs,be,f.size());
    return f;
}

// =============================================================================
// RESULT STRUCTURES — matches SEFI_simulate.cc
// =============================================================================
struct TransientResult {
    string image_name, mode;
    int    ground_truth_class=-1; string ground_truth_name;
    int    baseline_class=-1;     string baseline_name; float baseline_prob=0;
    int    faulty_class[TOP_K];   string faulty_name[TOP_K];
    float  faulty_prob[TOP_K];
    bool   correctly_classified=false; float prob_drop=0;
    size_t bytes_corrupted=0, bits_corrupted=0;
    string pieces_affected;
    bool   crash=false;
    TransientResult(){ for(int i=0;i<TOP_K;i++){faulty_class[i]=-1;faulty_prob[i]=0;} }
};

// Per-piece injection detail (for the new per-layer CSV)
struct LayerDetail {
    string image_name, piece_name;
    int    piece_idx;
    size_t conceptual_start, conceptual_end;
    uint64_t weight_phys;
    size_t   bytes_in_piece, bits_in_piece;
};

struct Metrics { float precision,recall,f1; };
static Metrics compute_metrics(const vector<TransientResult>&R,bool faulty){
    map<int,int>tp,fp,fn;
    for(auto&r:R){
        if(r.crash)continue;
        int gt=r.ground_truth_class, pred=faulty?r.faulty_class[0]:r.baseline_class;
        if(gt<0||pred<0)continue;
        if(tp.find(gt)==tp.end()){tp[gt]=0;fn[gt]=0;}
        if(fp.find(pred)==fp.end())fp[pred]=0;
        if(pred==gt)tp[gt]++;else{fn[gt]++;fp[pred]++;}
    }
    float sp=0,sr=0,n=0;
    for(auto&[cls,tpc]:tp){
        int fpc=fp.count(cls)?fp[cls]:0,fnc=fn.count(cls)?fn[cls]:0;
        float p=(tpc+fpc)>0?(float)tpc/(tpc+fpc):0.f;
        float r=(tpc+fnc)>0?(float)tpc/(tpc+fnc):0.f;
        sp+=p;sr+=r;n++;
    }
    float pr=n>0?sp/n:0.f,rc=n>0?sr/n:0.f;
    return{pr,rc,(pr+rc)>0?2*pr*rc/(pr+rc):0.f};
}

// =============================================================================
// CSV OUTPUT
// =============================================================================
static void write_results_csv(const vector<TransientResult>&results,
                               const string&out_dir,const char*mname,
                               const vector<string>&kinds){
    string path=out_dir+"/results_"+mname+".csv";
    ofstream f(path);if(!f)return;
    f<<"image_name,sefi_mode,target,transient,"
      "ground_truth_class,ground_truth_name,"
      "baseline_class,baseline_name,baseline_prob,";
    for(int i=0;i<TOP_K;i++)
        f<<"faulty_top"<<i+1<<",faulty_top"<<i+1<<"_name,faulty_top"<<i+1<<"_prob,";
    f<<"correctly_classified,prob_drop,"
      "bytes_corrupted,bits_corrupted,pieces_affected,crash\n";
    auto q=[](const string&s){return s.find(',')!=string::npos?"\""+s+"\"":s;};
    for(auto&R:results){
        f<<q(R.image_name)<<","<<R.mode<<",weights,1,"
         <<R.ground_truth_class<<","<<q(R.ground_truth_name)<<","
         <<R.baseline_class<<","<<q(R.baseline_name)<<","<<R.baseline_prob<<",";
        for(int i=0;i<TOP_K;i++)
            f<<R.faulty_class[i]<<","<<q(R.faulty_name[i])<<","<<R.faulty_prob[i]<<",";
        f<<(R.correctly_classified?1:0)<<","<<R.prob_drop<<","
         <<R.bytes_corrupted<<","<<R.bits_corrupted<<","
         <<q(R.pieces_affected)<<","<<(R.crash?1:0)<<"\n";
    }
    printf("[CSV] Saved: %s\n",path.c_str());
}

static void write_accuracy_csv(const string&out_dir,const char*mname,
        int total,int base_correct,float base_pct,Metrics bm,
        int faulty_correct,int faulty_wrong,float faulty_pct,Metrics fm){
    string path=out_dir+"/accuracy_summary.csv";
    ofstream f(path);if(!f)return;
    f<<"sefi_mode,total_images,"
      "baseline_correct,baseline_accuracy_pct,baseline_precision,baseline_recall,baseline_f1,"
      "faulty_correct,faulty_wrong,faulty_accuracy_pct,faulty_precision,faulty_recall,faulty_f1\n";
    f<<fixed<<setprecision(4)
     <<mname<<","<<total<<","
     <<base_correct<<","<<base_pct<<","<<bm.precision<<","<<bm.recall<<","<<bm.f1<<","
     <<faulty_correct<<","<<faulty_wrong<<","
     <<faulty_pct<<","<<fm.precision<<","<<fm.recall<<","<<fm.f1<<"\n";
    printf("[CSV] Saved: %s\n",path.c_str());
}

static void write_layer_details_csv(const vector<LayerDetail>&details,
                                     const string&out_dir,const char*mname){
    string path=out_dir+"/per_layer_details_"+string(mname)+".csv";
    ofstream f(path);if(!f)return;
    f<<"image_name,piece_name,piece_idx,"
      "conceptual_start,conceptual_end,"
      "weight_phys,bytes_in_piece,bits_in_piece\n";
    for(auto&d:details){
        f<<d.image_name<<","<<d.piece_name<<","<<d.piece_idx<<","
         <<d.conceptual_start<<","<<d.conceptual_end<<","
         <<"0x"<<hex<<d.weight_phys<<dec<<","
         <<d.bytes_in_piece<<","<<d.bits_in_piece<<"\n";
    }
    printf("[CSV] Saved: %s\n",path.c_str());
}

// =============================================================================
// MAIN
// =============================================================================
int main(int argc,char*argv[]) {
    if(argc<2){
        printf("Usage: %s <models_dir> [row|column|block] [-v]\n",argv[0]);return -1;
    }
    string models_dir=argv[1];
    TransientMode mode=TransientMode::ROW;
    size_t col_width=DDR4_COL_DEFAULT, block_size=4096;
    for(int i=2;i<argc;i++){
        string a=argv[i];
        if(a=="row")   mode=TransientMode::ROW;
        if(a=="column")mode=TransientMode::COLUMN;
        if(a=="block") mode=TransientMode::BLOCK;
    }
    if(mode==TransientMode::COLUMN){
        printf("Column band width in bytes [default %zu]: ",DDR4_COL_DEFAULT);fflush(stdout);
        char buf[64];if(fgets(buf,sizeof(buf),stdin))col_width=atol(buf);
        if(col_width==0)col_width=DDR4_COL_DEFAULT;
    }
    if(mode==TransientMode::BLOCK){
        printf("Block size in bytes [default 4096]: ");fflush(stdout);
        char buf[64];if(fgets(buf,sizeof(buf),stdin))block_size=atol(buf);
        if(block_size==0)block_size=4096;
    }
    string val_folder="./train_subset";
    printf("Image folder path [default %s]: ",val_folder.c_str());fflush(stdout);
    {char buf[512];if(fgets(buf,sizeof(buf),stdin)&&buf[0]!='\n'){
        buf[strcspn(buf,"\n")]=0;if(buf[0])val_folder=buf;}}

    g_devmem_fd=open("/dev/mem",O_RDWR|O_SYNC);
    if(g_devmem_fd<0){perror("open /dev/mem");return -1;}
    mt19937 rng(time(nullptr)^getpid());

    string out_dir="./FaultResults/sefi_transient_results/"+mode_folder(mode)+"/weights";
    mkdirp(out_dir);clear_dir(out_dir);
    string log_dir="./FaultResults/sefi_transient_results/"+mode_folder(mode);
    mkdirp(log_dir);
    g_logfp=fopen((log_dir+"/sefi_transient.log").c_str(),"w");

    // Header (terminal + log)
    sim_log("╔══════════════════════════════════════════════════════════╗\n");
    sim_log("║   SEFI Transient — Layer-by-Layer Weight Injection        ║\n");
    sim_log("║   ZCU104 | DPUCZDX8G | PyTorch ResNet50 (18-piece split)  ║\n");
    sim_log("╚══════════════════════════════════════════════════════════╝\n\n");
    sim_log("[Config] mode=%-22s  models=%s\n",mode_name(mode),models_dir.c_str());
    sim_log("[Config] images=%-22s  col_width=%zu  block_size=%zu\n\n",
            val_folder.c_str(),col_width,block_size);

    // ── Labels & images ──────────────────────────────────────────────────────
    vector<string>kinds;LoadWords("./words.txt",kinds);
    auto syn=LoadSynsets("./synset.txt");
    vector<ImageEntry>entries;ListImagesWithGroundTruth(val_folder,syn,entries);
    if(entries.empty()){fprintf(stderr,"No images in %s\n",val_folder.c_str());return -1;}
    sim_log("[Dataset] %zu images\n\n",entries.size());

    // ── Load graphs ───────────────────────────────────────────────────────────
    vector<PieceInfo>pieces(N_PIECES);
    size_t coff=0,total=0;
    for(int k=0;k<N_PIECES;k++)total+=PIECE_WEIGHT_TABLE[k].weight_size;

    for(int k=0;k<N_PIECES;k++){
        auto&p=pieces[k];
        p.name=PIECE_WEIGHT_TABLE[k].name;
        p.xmodel_path=models_dir+"/"+p.name+"/"+p.name+".xmodel";
        p.weight_size=PIECE_WEIGHT_TABLE[k].weight_size;
        p.conceptual_start=coff;coff+=p.weight_size;
        p.graph=xir::Graph::deserialize(p.xmodel_path);
        if(!p.graph){fprintf(stderr,"Cannot load %s\n",p.xmodel_path.c_str());return -1;}
        auto dpu=get_dpu_subgraph(p.graph.get());
        if(dpu.empty()){fprintf(stderr,"No DPU subgraph:%s\n",p.name.c_str());return -1;}
        p.dpu_subgraph=dpu[0];
    }

    // Discover addresses — log only, not terminal
    log_only("[Discover] Piece addresses and scales:\n");
    for(int k=0;k<N_PIECES;k++){
        discover_piece(pieces[k]);
        log_only("  [%02d] %-30s  w_phys=0x%lX  w_size=%zu  in_sc=%.5f  out_sc=%.5f\n",
                 k,pieces[k].name.c_str(),pieces[k].weight_phys,pieces[k].weight_size,
                 pieces[k].in_scale,pieces[k].out_scale);
    }
    log_only("  total_conceptual=%zu B\n\n",total);
    // Scale ratios — log only
    log_only("[ScaleCheck] Inter-piece ratios:\n");
    for(int k=0;k<N_PIECES-1;k++){
        float r=pieces[k].out_scale*pieces[k+1].in_scale;
        log_only("  piece_%02d→%02d ratio=%.5f%s\n",k,k+1,r,
                 fabs(r-1.f)>0.05f?" [requantize]":"");
    }
    log_only("\n");

    // ── Preprocess all images ────────────────────────────────────────────────
    int inH=224,inW=224;
    float in_scale=pieces[0].in_scale;
    int   in_elems=pieces[0].in_elems;
    int   outSz=pieces[N_PIECES-1].out_elems;
    float out_scale=pieces[N_PIECES-1].out_scale;

    vector<vector<int8_t>>imgBufs(entries.size());
    for(size_t i=0;i<entries.size();i++){
        Mat raw=imread(entries[i].path);if(raw.empty())continue;
        imgBufs[i].resize(in_elems);
        preprocess_image(raw,imgBufs[i].data(),inH,inW,in_scale);
    }

    // ── Baseline ─────────────────────────────────────────────────────────────
    sim_log("[Baseline] Running clean inference on %zu images ...\n",entries.size());
    struct BL{int cls=-1;float prob=0;string name;bool valid=false;};
    vector<BL>baselines(entries.size());
    int base_correct=0;

    for(size_t i=0;i<entries.size();i++){
        if(imgBufs[i].empty())continue;
        int8_t*cur=imgBufs[i].data();bool ok=true;
        for(int k=0;k<N_PIECES&&ok;k++){
            ok=exec_piece(pieces[k],k,cur,pieces[k].out_buf.data(),nullptr);
            requantize(pieces,k);
            cur=pieces[k].out_buf.data();
        }
        if(!ok)continue;
        vector<float>sm(outSz);
        CPUCalcSoftmax(pieces[N_PIECES-1].out_buf.data(),outSz,sm.data(),out_scale);
        auto tk=topk(sm.data(),outSz,1);
        baselines[i]={tk[0],sm[tk[0]],
            (tk[0]<(int)kinds.size())?kinds[tk[0]]:"?",true};
        if(tk[0]==entries[i].ground_truth)base_correct++;
        printf("  [baseline %zu/%zu]\r",i+1,entries.size());fflush(stdout);
    }
    float base_pct=entries.size()>0?100.f*base_correct/entries.size():0.f;
    printf("\n");  // clear \r progress line
    sim_log("[Baseline] Clean accuracy: %d/%zu = %.2f%%\n\n",
            base_correct,entries.size(),base_pct);

    // ── SEFI injection ────────────────────────────────────────────────────────
    sim_log("[SEFI] Starting %s on target=weights ...\n",mode_name(mode));
    vector<TransientResult>results;
    vector<LayerDetail>layer_details;
    int total_correct=0;

    for(size_t i=0;i<entries.size();i++){
        if(!baselines[i].valid||imgBufs[i].empty())continue;

        vector<ByteFlip>flips;
        switch(mode){
            case TransientMode::ROW:   flips=plan_row(rng,pieces,total);break;
            case TransientMode::COLUMN:flips=plan_column(rng,col_width,pieces);break;
            case TransientMode::BLOCK: flips=plan_block(rng,block_size,pieces,total);break;
        }

        // Build per-piece flip counts for layer details CSV
        map<int,pair<size_t,size_t>> piece_bytes_bits; // piece_id → (bytes, bits)
        for(auto&f:flips){
            piece_bytes_bits[f.piece_id].first++;
            piece_bytes_bits[f.piece_id].second+=__builtin_popcount(f.xor_mask);
        }

        int8_t*cur=imgBufs[i].data();bool crashed=false;
        set<int>affected;for(auto&f:flips)affected.insert(f.piece_id);

        for(int k=0;k<N_PIECES&&!crashed;k++){
            crashed=!exec_piece(pieces[k],k,cur,pieces[k].out_buf.data(),
                                affected.count(k)?&flips:nullptr);
            requantize(pieces,k);
            cur=pieces[k].out_buf.data();

            // Record layer detail if this piece was affected
            if(piece_bytes_bits.count(k)){
                LayerDetail ld;
                ld.image_name=entries[i].name;
                ld.piece_name=pieces[k].name;
                ld.piece_idx=k;
                ld.conceptual_start=pieces[k].conceptual_start;
                ld.conceptual_end=pieces[k].conceptual_start+pieces[k].weight_size;
                ld.weight_phys=pieces[k].weight_phys;
                ld.bytes_in_piece=piece_bytes_bits[k].first;
                ld.bits_in_piece=piece_bytes_bits[k].second;
                layer_details.push_back(ld);
            }
        }

        TransientResult R;
        R.image_name=entries[i].name; R.mode=mode_name(mode);
        R.ground_truth_class=entries[i].ground_truth;
        R.ground_truth_name=(R.ground_truth_class<(int)kinds.size())?
            kinds[R.ground_truth_class]:"?";
        R.baseline_class=baselines[i].cls;
        R.baseline_name=baselines[i].name;
        R.baseline_prob=baselines[i].prob;
        R.crash=crashed;

        if(!crashed){
            vector<float>sm(outSz);
            CPUCalcSoftmax(pieces[N_PIECES-1].out_buf.data(),outSz,sm.data(),out_scale);
            auto tk=topk(sm.data(),outSz,TOP_K);
            for(int t=0;t<TOP_K;t++){
                R.faulty_class[t]=tk[t];
                R.faulty_prob[t]=sm[tk[t]];
                R.faulty_name[t]=(tk[t]<(int)kinds.size())?kinds[tk[t]]:"?";
            }
            R.correctly_classified=(tk[0]==R.ground_truth_class);
            R.prob_drop=R.baseline_prob-R.faulty_prob[0];
        }
        for(auto&f:flips){R.bytes_corrupted++;R.bits_corrupted+=__builtin_popcount(f.xor_mask);}
        string aff;
        for(int pid:affected){if(!aff.empty())aff+=",";aff+=pieces[pid].name;}
        R.pieces_affected=aff.empty()?"none":aff;
        if(R.correctly_classified)total_correct++;
        results.push_back(R);

        // Per-image progress on terminal (no conceptual range)
        sim_log("  [%zu/%zu] %-45s  [%s]  bytes=%zu  bits~%zu\n",
                i+1,entries.size(),entries[i].name.c_str(),
                mode_name(mode),R.bytes_corrupted,R.bits_corrupted);
        if(!affected.empty())
            sim_log("  [%s] pieces=%s\n",mode_name(mode),R.pieces_affected.c_str());

        // Full detail to log file only
        log_only("  detail: top1=%s  drop=%.3f%s\n",
                 R.faulty_name[0].c_str(),R.prob_drop,R.crash?" CRASH":"");
    }

    // ── Summary & output ──────────────────────────────────────────────────────
    float faulty_pct=results.size()>0?100.f*total_correct/results.size():0.f;
    Metrics bm=compute_metrics(results,false);
    Metrics fm=compute_metrics(results,true);

    sim_log("\n[Summary] target=%-14s  baseline=%.2f%%  faulty=%.2f%%"
            "  P=%.3f  R=%.3f  F1=%.3f\n",
            "weights",base_pct,faulty_pct,fm.precision,fm.recall,fm.f1);
    log_only("[Summary] baseline P=%.3f R=%.3f F1=%.3f\n",bm.precision,bm.recall,bm.f1);

    write_results_csv(results,out_dir,mode_name(mode),kinds);
    write_accuracy_csv(out_dir,mode_name(mode),
        (int)results.size(),base_correct,base_pct,bm,
        total_correct,(int)results.size()-total_correct,faulty_pct,fm);
    write_layer_details_csv(layer_details,out_dir,mode_name(mode));

    if(g_logfp)fclose(g_logfp);
    close(g_devmem_fd);
    return 0;
}
