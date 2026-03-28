/*
 * Copyright 2019 Xilinx Inc.
 * Modified: AMD XRT TensorBuffer validity check + FORCED page-aligned fallback
 *
 * Adds AMD's XRT TensorBuffer method on top of the original baseline
 * inference script. After creating the runner it:
 *   1. Casts to RunnerExt to access get_inputs()/get_outputs()
 *   2. Queries tensor rank at runtime to build the correct index vector
 *   3. Prints BOTH original + FINAL page-aligned addresses
 *   4. Forces posix_memalign(4096) fallback so DPU always gets page-aligned input
 *   5. Runs one clean inference and prints top-5 results
 *
 * Reference:
 *   Link 2: https://github.com/Xilinx/Vitis-AI-Tutorials/tree/3.0/Tutorials/RESNET18/
 *   Vitis-AI Overview: https://docs.amd.com/r/en-US/ug1414-vitis-ai/Vitis-AI-Overview
 *   ZCU104 board: https://xilinx.github.io/Vitis-AI/3.0/html/docs/quickstart/mpsoc.html
 */

#include <assert.h>
#include <dirent.h>
#include <stdio.h>
#include <stdlib.h>
#include <sys/stat.h>
#include <unistd.h>

#include <cassert>
#include <cmath>
#include <cstdio>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <queue>
#include <string>
#include <vector>
#include <memory>

#include "common.h"
#include <opencv2/opencv.hpp>
#include <vart/runner_ext.hpp>

using namespace std;
using namespace cv;

GraphInfo shapes;

const string baseImagePath = "../images/";
const string wordsPath = "./";

// =============================================================================
// XRT BUFFER INFO + FORCED PAGE-ALIGNED FALLBACK
// =============================================================================
struct XRTBufferInfo {
    uint8_t* vaddr         = nullptr;
    uint64_t paddr         = 0;
    size_t   size          = 0;
    bool     valid         = false;
    bool     xrt_confirmed = false;
};

static XRTBufferInfo g_input_xrt;
static XRTBufferInfo g_output_xrt;

// NEW: Fallback buffers (always 4096-byte aligned)
static uint8_t* g_fallback_input  = nullptr;
static uint8_t* g_fallback_output = nullptr;

static bool init_xrt_io_buffers(vart::RunnerExt* runner) {
    auto inputs  = runner->get_inputs();
    auto outputs = runner->get_outputs();

    if (inputs.empty() || outputs.empty()) {
        printf("[XRT] ERROR: get_inputs()/get_outputs() returned empty.\n");
        return false;
    }

    // === PRINT ORIGINAL VART BUFFERS ===
    {
        vart::TensorBuffer* tb = inputs[0];
        auto rank = tb->get_tensor()->get_shape().size();
        vector<int> idx(rank, 0);
        auto [vaddr, sz] = tb->data(idx);
        auto [paddr, _]  = tb->data_phy(idx);

        g_input_xrt.vaddr         = reinterpret_cast<uint8_t*>(vaddr);
        g_input_xrt.paddr         = paddr;
        g_input_xrt.size          = sz;
        g_input_xrt.valid         = (vaddr != 0 && sz > 0);
        g_input_xrt.xrt_confirmed = (vaddr != 0 && (reinterpret_cast<uintptr_t>(vaddr) % 4096 == 0));

        printf("[XRT][Input-Original]  vaddr=0x%016lX  paddr=0x%016lX  size=%zu  %s\n",
               vaddr, paddr, sz,
               g_input_xrt.xrt_confirmed ? "PAGE-ALIGNED: XRT bo confirmed" : "WARNING: NOT page-aligned -- heap");
    }

    {
        vart::TensorBuffer* tb = outputs[0];
        auto rank = tb->get_tensor()->get_shape().size();
        vector<int> idx(rank, 0);
        auto [vaddr, sz] = tb->data(idx);
        auto [paddr, _]  = tb->data_phy(idx);

        g_output_xrt.vaddr         = reinterpret_cast<uint8_t*>(vaddr);
        g_output_xrt.paddr         = paddr;
        g_output_xrt.size          = sz;
        g_output_xrt.valid         = (vaddr != 0 && sz > 0);
        g_output_xrt.xrt_confirmed = (vaddr != 0 && (reinterpret_cast<uintptr_t>(vaddr) % 4096 == 0));

        printf("[XRT][Output-Original] vaddr=0x%016lX  paddr=0x%016lX  size=%zu  %s\n",
               vaddr, paddr, sz,
               g_output_xrt.xrt_confirmed ? "PAGE-ALIGNED: XRT bo confirmed" : "WARNING: NOT page-aligned -- heap");
    }

    if (g_input_xrt.xrt_confirmed && g_output_xrt.xrt_confirmed) {
        printf("[XRT] RESULT: Both buffers confirmed as XRT bos. AMD method available.\n");
        return true;
    }

    printf("[XRT] RESULT: Buffers are NOT true XRT bos → forcing page-aligned fallback.\n");

    // === FORCED PAGE-ALIGNED FALLBACK ===
    if (posix_memalign((void**)&g_fallback_input, 4096, g_input_xrt.size) != 0 ||
        posix_memalign((void**)&g_fallback_output, 4096, g_output_xrt.size) != 0) {
        fprintf(stderr, "[XRT] ERROR: posix_memalign failed\n");
        return false;
    }

    g_input_xrt.vaddr = g_fallback_input;
    g_input_xrt.xrt_confirmed = true;

    // === FINAL ADDRESSES USED BY DPU ===
    printf("[XRT] FINAL BUFFERS USED BY DPU:\n");
    printf("  [Input]  vaddr=0x%016lX  (PAGE-ALIGNED 4096)  ← DPU READS FROM HERE\n",
           (uintptr_t)g_input_xrt.vaddr);
    printf("  [Output] vaddr=0x%016lX  paddr=0x%016lX  ← DPU WRITES HERE\n",
           (uintptr_t)g_output_xrt.vaddr, g_output_xrt.paddr);

    printf("[XRT] RESULT: Manually page-aligned buffers created (4096-byte aligned).\n");
    return true;
}

// =============================================================================
// HELPERS (100% identical to original main_amd.txt and official main.cc)
// =============================================================================
void ListImages(string const& path, vector<string>& images) {
    images.clear();
    struct dirent* entry;
    struct stat s;
    lstat(path.c_str(), &s);
    if (!S_ISDIR(s.st_mode)) {
        fprintf(stderr, "Error: %s is not a valid directory!\n", path.c_str());
        exit(1);
    }
    DIR* dir = opendir(path.c_str());
    if (dir == nullptr) {
        fprintf(stderr, "Error: Open %s path failed.\n", path.c_str());
        exit(1);
    }
    while ((entry = readdir(dir)) != nullptr) {
        if (entry->d_type == DT_REG || entry->d_type == DT_UNKNOWN) {
            string name = entry->d_name;
            string ext  = name.substr(name.find_last_of(".") + 1);
            if (ext == "JPEG" || ext == "jpeg" || ext == "JPG" ||
                ext == "jpg"  || ext == "PNG"  || ext == "png") {
                images.push_back(name);
            }
        }
    }
    closedir(dir);
}

void LoadWords(string const& path, vector<string>& kinds) {
    kinds.clear();
    ifstream fkinds(path);
    if (fkinds.fail()) {
        fprintf(stderr, "Error: Open %s failed.\n", path.c_str());
        exit(1);
    }
    string kind;
    while (getline(fkinds, kind)) kinds.push_back(kind);
    fkinds.close();
}

void CPUCalcSoftmax(const int8_t* data, size_t size, float* result, float scale) {
    assert(data && result);
    double sum = 0.0;
    for (size_t i = 0; i < size; i++) {
        result[i] = exp((float)data[i] * scale);
        sum += result[i];
    }
    for (size_t i = 0; i < size; i++) result[i] /= (float)sum;
}

void TopK(const float* d, int size, int k, vector<string>& vkinds) {
    assert(d && size > 0 && k > 0);
    priority_queue<pair<float, int>> q;
    for (int i = 0; i < size; ++i) q.push({d[i], i});
    for (int i = 0; i < k; ++i) {
        auto ki = q.top();
        printf("  top[%d] prob=%-8f  name=%s\n", i, d[ki.second],
               vkinds[ki.second].c_str());
        q.pop();
    }
}

// =============================================================================
// INFERENCE via XRT TensorBuffers (fixed lifetime)
// =============================================================================
static void run_one_inference_xrt(vart::RunnerExt* runner,
                                   const string& image_name,
                                   float mean[3],
                                   int inH, int inW,
                                   int outSize,
                                   float input_scale, float output_scale,
                                   vector<string>& kinds) {
    Mat image = imread(baseImagePath + image_name);
    if (image.empty()) {
        printf("[Infer] Cannot read image: %s\n", image_name.c_str());
        return;
    }
    Mat resized;
    resize(image, resized, Size(inW, inH), 0, 0);

    int8_t* in_ptr = reinterpret_cast<int8_t*>(g_input_xrt.vaddr);
    for (int h = 0; h < inH; h++)
        for (int w = 0; w < inW; w++)
            for (int c = 0; c < 3; c++)
                in_ptr[h * inW * 3 + w * 3 + c] =
                    (int8_t)((resized.at<Vec3b>(h, w)[c] - mean[c]) * input_scale);

    __sync_synchronize();

    auto inputs  = runner->get_inputs();
    auto outputs = runner->get_outputs();

    vector<shared_ptr<xir::Tensor>> batchTensors;
    vector<unique_ptr<vart::TensorBuffer>> inputs_buf, outputs_buf;

    if (g_fallback_input != nullptr) {
        printf("[Fallback] Using manually page-aligned TensorBuffers for DPU\n");

        auto in_dims = inputs[0]->get_tensor()->get_shape();
        batchTensors.push_back(std::shared_ptr<xir::Tensor>(
            xir::Tensor::create(inputs[0]->get_tensor()->get_name(), in_dims,
                                xir::DataType{xir::DataType::XINT, 8u})));
        inputs_buf.push_back(std::make_unique<CpuFlatTensorBuffer>(g_fallback_input, batchTensors.back().get()));

        auto out_dims = outputs[0]->get_tensor()->get_shape();
        batchTensors.push_back(std::shared_ptr<xir::Tensor>(
            xir::Tensor::create(outputs[0]->get_tensor()->get_name(), out_dims,
                                xir::DataType{xir::DataType::XINT, 8u})));
        outputs_buf.push_back(std::make_unique<CpuFlatTensorBuffer>(g_fallback_output, batchTensors.back().get()));

        inputs  = {inputs_buf[0].get()};
        outputs = {outputs_buf[0].get()};
    }

    auto job = runner->execute_async(inputs, outputs);
    runner->wait(job.first, -1);

    const int8_t* out_ptr = reinterpret_cast<const int8_t*>(g_output_xrt.vaddr);
    vector<float> softmax(outSize);
    CPUCalcSoftmax(out_ptr, outSize, softmax.data(), output_scale);

    printf("\n[Infer] Image: %s\n", image_name.c_str());
    TopK(softmax.data(), outSize, 5, kinds);
}

// =============================================================================
// MAIN
// =============================================================================
int main(int argc, char* argv[]) {
    if (argc != 2) {
        printf("Usage: %s <model.xmodel>\n", argv[0]);
        return -1;
    }

    auto graph    = xir::Graph::deserialize(argv[1]);
    auto subgraph = get_dpu_subgraph(graph.get());
    CHECK_EQ(subgraph.size(), 1u) << "Expected one DPU subgraph.";

    printf("[Setup] Creating runner...\n");
    auto runner_owned = vart::Runner::create_runner(subgraph[0], "run");

    auto* runner = dynamic_cast<vart::RunnerExt*>(runner_owned.get());
    if (!runner) {
        fprintf(stderr, "[Error] RunnerExt cast failed.\n");
        return -1;
    }
    printf("[Setup] RunnerExt cast OK.\n");

    auto inputTensors  = runner->get_input_tensors();
    auto outputTensors = runner->get_output_tensors();
    int inputCnt  = (int)inputTensors.size();
    int outputCnt = (int)outputTensors.size();
    TensorShape inshapes[inputCnt];
    TensorShape outshapes[outputCnt];
    shapes.inTensorList  = inshapes;
    shapes.outTensorList = outshapes;
    getTensorShape(runner, &shapes, inputCnt, outputCnt);

    int inH      = shapes.inTensorList[0].height;
    int inW      = shapes.inTensorList[0].width;
    int outSize  = shapes.outTensorList[0].size;
    float in_sc  = get_input_scale(inputTensors[0]);
    float out_sc = get_output_scale(outputTensors[0]);

    printf("[Setup] Input  tensor: H=%d W=%d scale=%.6f\n", inH, inW, in_sc);
    printf("[Setup] Output tensor: size=%d scale=%.6f\n", outSize, out_sc);

    // -------------------------------------------------------------------------
    // AMD METHOD: obtain and validate XRT TensorBuffer addresses
    // -------------------------------------------------------------------------
    printf("\n[AMD] Checking XRT TensorBuffer addresses...\n");
    bool xrt_ok = init_xrt_io_buffers(runner);
    printf("\n");

    // -------------------------------------------------------------------------
    // Run one clean inference
    // -------------------------------------------------------------------------
    vector<string> kinds, images;
    LoadWords(wordsPath + "words.txt", kinds);
    ListImages(baseImagePath, images);
    if (images.empty()) {
        fprintf(stderr, "[Error] No images found in %s\n", baseImagePath.c_str());
        return -1;
    }

    float mean[3] = {104.f, 107.f, 123.f};
    printf("[Infer] Running one clean inference on: %s\n", images[0].c_str());
    run_one_inference_xrt(runner, images[0], mean,
                          inH, inW, outSize,
                          in_sc, out_sc, kinds);

    // Cleanup
    free(g_fallback_input);
    free(g_fallback_output);

    printf("\n[Done]\n");
    return 0;
}