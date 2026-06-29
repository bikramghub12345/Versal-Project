/*
 * controlRegisters.cc -- DPU Control Register Reader
 * ====================================================
 * Prints all DPUCZDX8G AXI slave register values for both cores
 * during inference. No interpretation -- raw addresses only.
 *
 * Register offsets from xclbinutil (dpu.xclbin):
 *   0x40  dpu_doneclr
 *   0x44  dpu_prof_en
 *   0x48  dpu_cmd
 *   0x50  dpu_instr_addr   (M_AXI_GP0, HPC0)
 *   0x58  dpu_prof_addr    (M_AXI_GP0, HPC0)
 *   0x60  dpu_base0_addr   (M_AXI_HP0)
 *   0x68  dpu_base1_addr   (M_AXI_HP0)
 *   0x70  dpu_base2_addr   (M_AXI_HP0)
 *   0x78  dpu_base3_addr   (M_AXI_HP0)
 *   0x80  dpu_base4_addr   (M_AXI_HP2)
 *   0x88  dpu_base5_addr   (M_AXI_HP2)
 *   0x90  dpu_base6_addr   (M_AXI_HP2)
 *   0x98  dpu_base7_addr   (M_AXI_HP2)
 *
 * BUILD:
 *   g++ -std=c++17 -O2 -o controlRegisters src/controlRegisters.cc \
 *       ../common/common.cpp \
 *       -I ./src -I ../common \
 *       -I /usr/include/opencv4 -I /usr/include/vitis_ai \
 *       $(pkg-config --cflags --libs opencv4) \
 *       -lvart-runner -lxir -lglog -lpthread
 *
 * RUN (must be root):
 *   ./controlRegisters <model.xmodel>
 */

#include <fcntl.h>
#include <stdio.h>
#include <stdlib.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>
#include <dirent.h>

#include <algorithm>
#include <chrono>
#include <cstring>
#include <iomanip>
#include <iostream>
#include <memory>
#include <string>
#include <thread>
#include <vector>

#include "common.h"
#include <opencv2/opencv.hpp>

using namespace std;
using namespace cv;
using namespace std::chrono;

// =============================================================================
// REGISTER TABLE  (offsets from xclbinutil, no interpretation)
// =============================================================================
struct DpuReg {
    uint32_t    offset;
    const char* name;
    const char* port;      // from xclbin: which AXI port
    const char* memory;    // from xclbin: which DDR4 bank (core1 / core2)
};

static const DpuReg REGS[] = {
    { 0x40, "dpu_doneclr",   "S_AXI_CONTROL", "-"          },
    { 0x44, "dpu_prof_en",   "S_AXI_CONTROL", "-"          },
    { 0x48, "dpu_cmd",       "S_AXI_CONTROL", "-"          },
    { 0x50, "dpu_instr_addr","M_AXI_GP0",     "HPC0"       },
    { 0x54, "dpu_instr_addr","M_AXI_GP0",     "HPC0 (H32)" },
    { 0x58, "dpu_prof_addr", "M_AXI_GP0",     "HPC0"       },
    { 0x5C, "dpu_prof_addr", "M_AXI_GP0",     "HPC0 (H32)" },
    { 0x60, "dpu_base0_addr","M_AXI_HP0",     "HP0/HP2"    },
    { 0x64, "dpu_base0_addr","M_AXI_HP0",     "HP0/HP2(H)" },
    { 0x68, "dpu_base1_addr","M_AXI_HP0",     "HP0/HP2"    },
    { 0x6C, "dpu_base1_addr","M_AXI_HP0",     "HP0/HP2(H)" },
    { 0x70, "dpu_base2_addr","M_AXI_HP0",     "HP0/HP2"    },
    { 0x74, "dpu_base2_addr","M_AXI_HP0",     "HP0/HP2(H)" },
    { 0x78, "dpu_base3_addr","M_AXI_HP0",     "HP0/HP2"    },
    { 0x7C, "dpu_base3_addr","M_AXI_HP0",     "HP0/HP2(H)" },
    { 0x80, "dpu_base4_addr","M_AXI_HP2",     "HP1/HP3"    },
    { 0x84, "dpu_base4_addr","M_AXI_HP2",     "HP1/HP3(H)" },
    { 0x88, "dpu_base5_addr","M_AXI_HP2",     "HP1/HP3"    },
    { 0x8C, "dpu_base5_addr","M_AXI_HP2",     "HP1/HP3(H)" },
    { 0x90, "dpu_base6_addr","M_AXI_HP2",     "HP1/HP3"    },
    { 0x94, "dpu_base6_addr","M_AXI_HP2",     "HP1/HP3(H)" },
    { 0x98, "dpu_base7_addr","M_AXI_HP2",     "HP1/HP3"    },
    { 0x9C, "dpu_base7_addr","M_AXI_HP2",     "HP1/HP3(H)" },
};
static const int N_REGS = (int)(sizeof(REGS)/sizeof(REGS[0]));

struct DpuCore {
    const char* name;
    uint32_t    base;
};
static const DpuCore CORES[] = {
    { "DPUCZDX8G_1", 0x80000000 },
    { "DPUCZDX8G_2", 0x80001000 },
};
static const int N_CORES = 2;

// =============================================================================
// /dev/mem ACCESS
// =============================================================================
static volatile uint32_t* map_core(int fd, uint32_t base) {
    void* m = mmap(NULL, 4096, PROT_READ|PROT_WRITE, MAP_SHARED, fd, (off_t)base);
    if (m == MAP_FAILED) { perror("mmap"); return nullptr; }
    return (volatile uint32_t*)m;
}

static uint32_t rd(volatile uint32_t* base, uint32_t off) {
    return base[off/4];
}

// =============================================================================
// PRINT TABLE FOR ONE CORE
// =============================================================================
static void print_table(const DpuCore& c, volatile uint32_t* regs) {
    printf("\n  %s  (base=0x%08X)\n", c.name, c.base);
    printf("  %-12s  %-18s  %-14s  %-12s  %s\n",
           "Phys Addr", "Register", "Value (hex)", "Value (dec)", "AXI Port / DDR4 Bank");
    printf("  %s\n", string(80,'-').c_str());

    for (int i = 0; i < N_REGS; i++) {
        uint32_t val  = rd(regs, REGS[i].offset);
        uint32_t phys = c.base + REGS[i].offset;
        printf("  0x%08X  %-18s  0x%08X    %-12u  %s / %s\n",
               phys, REGS[i].name, val, val,
               REGS[i].port, REGS[i].memory);
    }
}

// =============================================================================
// MINIMAL INFERENCE SETUP
// =============================================================================
GraphInfo shapes;
static const string imgPath = "../images/";

static vector<string> list_images(const string& path) {
    vector<string> v;
    DIR* d = opendir(path.c_str()); if(!d) return v;
    struct dirent* e;
    while((e=readdir(d))) {
        string n=e->d_name; if(n.size()<4) continue;
        string ext=n.substr(n.find_last_of('.')+1);
        transform(ext.begin(),ext.end(),ext.begin(),::tolower);
        if(ext=="jpg"||ext=="jpeg"||ext=="png") v.push_back(n);
    }
    closedir(d); return v;
}

static void preprocess(const string& p, int8_t* dst,
                        int H, int W, float sc) {
    static const float mean[3]={104,107,123};
    Mat img=imread(p), rsz;
    if(img.empty()) return;
    resize(img,rsz,Size(W,H));
    for(int h=0;h<H;h++) for(int w=0;w<W;w++)
        for(int c=0;c<3;c++)
            dst[h*W*3+w*3+c]=(int8_t)((rsz.at<Vec3b>(h,w)[c]-mean[c])*sc);
}

// =============================================================================
// MAIN
// =============================================================================
int main(int argc, char* argv[]) {
    if(argc!=2){
        printf("Usage: %s <model.xmodel>\n",argv[0]);
        return -1;
    }

    int fd = open("/dev/mem", O_RDWR|O_SYNC);
    if(fd<0){ perror("open /dev/mem"); return -1; }

    volatile uint32_t* core_regs[N_CORES]={};
    for(int i=0;i<N_CORES;i++){
        core_regs[i] = map_core(fd, CORES[i].base);
        if(!core_regs[i]){ close(fd); return -1; }
    }

    // Load model + create runner
    auto graph    = xir::Graph::deserialize(argv[1]);
    auto subgraph = get_dpu_subgraph(graph.get());
    CHECK_EQ(subgraph.size(),1u) << "Expected one DPU subgraph";
    auto runner_owned = vart::Runner::create_runner(subgraph[0],"run");
    vart::Runner* runner = runner_owned.get();

    auto inT  = runner->get_input_tensors();
    auto outT = runner->get_output_tensors();
    TensorShape insh[8],outsh[8];
    shapes.inTensorList=insh; shapes.outTensorList=outsh;
    getTensorShape(runner,&shapes,(int)inT.size(),(int)outT.size());

    int   inSz  = shapes.inTensorList[0].size;
    int   inH   = shapes.inTensorList[0].height;
    int   inW   = shapes.inTensorList[0].width;
    int   outSz = shapes.outTensorList[0].size;
    float in_sc = get_input_scale(inT[0]);

    auto images = list_images(imgPath);
    if(images.empty()){ fprintf(stderr,"No images in %s\n",imgPath.c_str()); return -1; }

    vector<int8_t> imgBuf(inSz,0), fcBuf(outSz,0);
    preprocess(imgPath+images[0], imgBuf.data(), inH, inW, in_sc);

    auto idims=inT[0]->get_shape();  idims[0]=1;
    auto odims=outT[0]->get_shape(); odims[0]=1;
    vector<shared_ptr<xir::Tensor>> bt;
    bt.push_back(shared_ptr<xir::Tensor>(xir::Tensor::create(
        inT[0]->get_name(),idims,xir::DataType{xir::DataType::XINT,8u})));
    auto ib=make_unique<CpuFlatTensorBuffer>(imgBuf.data(),bt.back().get());
    bt.push_back(shared_ptr<xir::Tensor>(xir::Tensor::create(
        outT[0]->get_name(),odims,xir::DataType{xir::DataType::XINT,8u})));
    auto ob=make_unique<CpuFlatTensorBuffer>(fcBuf.data(),bt.back().get());
    vector<vart::TensorBuffer*> ip={ib.get()}, op={ob.get()};

    // Launch async, wait 5ms, read registers during execution
    auto job = runner->execute_async(ip, op);
    this_thread::sleep_for(milliseconds(5));

    // Print table only during inference
    printf("========================================================\n");
    printf("  DPU CONTROL REGISTERS -- DURING INFERENCE\n");
    printf("  Model: %s\n", argv[1]);
    printf("  Image: %s\n", images[0].c_str());
    printf("========================================================\n");

    for(int i=0;i<N_CORES;i++)
        print_table(CORES[i], core_regs[i]);

    runner->wait(job.first, -1);

    printf("\n[Done]\n");

    for(int i=0;i<N_CORES;i++) munmap((void*)core_regs[i],4096);
    close(fd);
    return 0;
}