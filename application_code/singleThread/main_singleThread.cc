/*
 * main_singleThread.cc
 *
 * Single-threaded ResNet50 inference over all images in a folder.
 * Processes images one at a time (batch=1), prints Top-5 for each.
 *
 * Usage: ./resnet50_st <model.xmodel> <image_folder>
 *
 * Based on the original main.cc (Xilinx Vitis-AI ResNet50 example).
 */

#include <assert.h>
#include <dirent.h>
#include <stdio.h>
#include <stdlib.h>
#include <sys/stat.h>
#include <unistd.h>

#include <algorithm>
#include <cassert>
#include <chrono>
#include <cmath>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <string>
#include <vector>

#include "common.h"
#include <opencv2/opencv.hpp>

using namespace std;
using namespace cv;
using namespace std::chrono;

GraphInfo shapes;

const string wordsPath = "./";

// ── Helpers (same as original main.cc) ───────────────────────────────────────

void ListImages(const string& path, vector<string>& images) {
    images.clear();
    struct stat s;
    lstat(path.c_str(), &s);
    if (!S_ISDIR(s.st_mode)) {
        fprintf(stderr, "Error: %s is not a valid directory!\n", path.c_str());
        exit(1);
    }
    DIR* dir = opendir(path.c_str());
    if (!dir) {
        fprintf(stderr, "Error: Cannot open directory %s\n", path.c_str());
        exit(1);
    }
    struct dirent* entry;
    while ((entry = readdir(dir)) != nullptr) {
        if (entry->d_type == DT_REG || entry->d_type == DT_UNKNOWN) {
            string name = entry->d_name;
            string ext  = name.substr(name.find_last_of('.') + 1);
            if (ext=="jpg"||ext=="jpeg"||ext=="JPG"||ext=="JPEG"||
                ext=="png"||ext=="PNG")
                images.push_back(name);
        }
    }
    closedir(dir);
    sort(images.begin(), images.end());  // deterministic order
}

void LoadWords(const string& path, vector<string>& kinds) {
    kinds.clear();
    ifstream f(path);
    if (f.fail()) {
        fprintf(stderr, "Error: Cannot open %s\n", path.c_str());
        exit(1);
    }
    string line;
    while (getline(f, line)) kinds.push_back(line);
}

void CPUCalcSoftmax(const int8_t* data, size_t size, float* result, float scale) {
    double sum = 0.0;
    for (size_t i = 0; i < size; i++) { result[i] = exp((float)data[i] * scale); sum += result[i]; }
    for (size_t i = 0; i < size; i++) result[i] /= sum;
}

void TopK(const float* d, int size, int k, const vector<string>& kinds) {
    vector<pair<float,int>> scores(size);
    for (int i = 0; i < size; i++) scores[i] = {d[i], i};
    sort(scores.begin(), scores.end(), greater<pair<float,int>>());
    for (int i = 0; i < k && i < size; i++)
        printf("  top[%d]  prob=%.6f  class=%d  %s\n",
               i, scores[i].first, scores[i].second,
               scores[i].second < (int)kinds.size()
                   ? kinds[scores[i].second].c_str() : "?");
}

// ── Main inference loop ───────────────────────────────────────────────────────

void runResnet50_singleThread(vart::Runner* runner,
                               const string& imageFolder,
                               const vector<string>& kinds) {
    vector<string> images;
    ListImages(imageFolder, images);
    if (images.empty()) {
        fprintf(stderr, "Error: No images found in %s\n", imageFolder.c_str());
        return;
    }
    printf("[Info] Found %zu images in %s\n\n", images.size(), imageFolder.c_str());

    // Tensor layout
    auto inputTensors  = runner->get_input_tensors();
    auto outputTensors = runner->get_output_tensors();

    float input_scale  = get_input_scale(inputTensors[0]);
    float output_scale = get_output_scale(outputTensors[0]);

    int outSize  = shapes.outTensorList[0].size;
    int inSize   = shapes.inTensorList[0].size;
    int inHeight = shapes.inTensorList[0].height;
    int inWidth  = shapes.inTensorList[0].width;

    // Fixed mean values for ResNet50 (BGR, same as original)
    const float mean[3] = {104.f, 107.f, 123.f};

    // Allocate buffers once, reuse for every image
    vector<int8_t> imageInputs(inSize);
    vector<int8_t> FCResult(outSize);
    vector<float>  softmax(outSize);

    size_t processed = 0;
    double total_img_ms = 0.0;

    auto wall_start = chrono::steady_clock::now();

    for (size_t n = 0; n < images.size(); n++) {
        const string imgPath = imageFolder + "/" + images[n];

        // ── Per-image timer starts here (includes imread) ─────────────────
        auto img_start = chrono::steady_clock::now();

        Mat image = imread(imgPath);
        if (image.empty()) {
            fprintf(stderr, "[Skip] Cannot read: %s\n", imgPath.c_str());
            continue;
        }

        // ── Preprocess ───────────────────────────────────────────────────
        Mat resized;
        resize(image, resized, Size(inWidth, inHeight));
        for (int h = 0; h < inHeight; h++)
            for (int w = 0; w < inWidth; w++)
                for (int c = 0; c < 3; c++)
                    imageInputs[h * inWidth * 3 + w * 3 + c] =
                        (int8_t)((resized.at<Vec3b>(h, w)[c] - mean[c]) * input_scale);

        // ── Build tensor buffers (batch=1) ───────────────────────────────
        auto in_dims  = inputTensors[0]->get_shape();  in_dims[0]  = 1;
        auto out_dims = outputTensors[0]->get_shape(); out_dims[0] = 1;

        vector<shared_ptr<xir::Tensor>> batchTensors;
        batchTensors.push_back(shared_ptr<xir::Tensor>(
            xir::Tensor::create(inputTensors[0]->get_name(), in_dims,
                                xir::DataType{xir::DataType::XINT, 8u})));
        batchTensors.push_back(shared_ptr<xir::Tensor>(
            xir::Tensor::create(outputTensors[0]->get_name(), out_dims,
                                xir::DataType{xir::DataType::XINT, 8u})));

        vector<unique_ptr<vart::TensorBuffer>> inputs, outputs;
        inputs.push_back(make_unique<CpuFlatTensorBuffer>(
            imageInputs.data(), batchTensors[0].get()));
        outputs.push_back(make_unique<CpuFlatTensorBuffer>(
            FCResult.data(), batchTensors[1].get()));

        vector<vart::TensorBuffer*> inputsPtr  = {inputs[0].get()};
        vector<vart::TensorBuffer*> outputsPtr = {outputs[0].get()};

        // ── Run DPU ──────────────────────────────────────────────────────
        auto job_id = runner->execute_async(inputsPtr, outputsPtr);
        runner->wait(job_id.first, -1);

        // ── Postprocess ──────────────────────────────────────────────────
        CPUCalcSoftmax(FCResult.data(), outSize, softmax.data(), output_scale);

        double img_ms = chrono::duration<double,milli>(
                            chrono::steady_clock::now() - img_start).count();
        total_img_ms += img_ms;
        processed++;

        printf("[%4zu / %4zu]  %s\n", n + 1, images.size(), images[n].c_str());
        printf("  time=%.2f ms\n", img_ms);
        TopK(softmax.data(), outSize, 5, kinds);
        printf("\n");
    }

    auto wall_end = chrono::steady_clock::now();

    // ── Summary ──────────────────────────────────────────────────────────────
    if (processed > 0) {
        double avg_ms   = total_img_ms / processed;
        double wall_ms  = chrono::duration<double,milli>(wall_end - wall_start).count();
        double fps      = 1000.0 / avg_ms;

        printf("════════════════════════════════════════\n");
        printf("  TIMING SUMMARY  (%zu images)\n", processed);
        printf("════════════════════════════════════════\n");
        printf("  Total time       : %8.2f ms  (%.3f s)\n", total_img_ms, total_img_ms/1000.0);
        printf("  Avg time/image   : %8.2f ms\n", avg_ms);
        printf("  FPS              : %8.2f  (= 1000 / %.2f ms)\n", fps, avg_ms);
        printf("════════════════════════════════════════\n");
        (void)wall_ms;  // wall_ms ≈ total_img_ms since imread is included
    }

    printf("\n[Done] Processed %zu / %zu images.\n", processed, images.size());
}

// ── Entry point ───────────────────────────────────────────────────────────────

int main(int argc, char* argv[]) {
    if (argc != 3) {
        printf("Usage: %s <model.xmodel> <image_folder>\n", argv[0]);
        printf("  model.xmodel  — compiled DPU model\n");
        printf("  image_folder  — directory containing jpg/png images\n");
        return 1;
    }

    const string modelPath   = argv[1];
    const string imageFolder = argv[2];

    // Load class labels
    vector<string> kinds;
    LoadWords(wordsPath + "words.txt", kinds);
    if (kinds.empty()) {
        fprintf(stderr, "Error: words.txt not found or empty in %s\n",
                wordsPath.c_str());
        return 1;
    }

    // Deserialize model and create runner
    auto graph    = xir::Graph::deserialize(modelPath);
    auto subgraph = get_dpu_subgraph(graph.get());
    if (subgraph.size() != 1u) {
        fprintf(stderr, "Error: Expected exactly 1 DPU subgraph, got %zu\n",
                subgraph.size());
        return 1;
    }

    auto runner = vart::Runner::create_runner(subgraph[0], "run");

    // Populate shapes
    auto inputTensors  = runner->get_input_tensors();
    auto outputTensors = runner->get_output_tensors();
    int inputCnt  = (int)inputTensors.size();
    int outputCnt = (int)outputTensors.size();
    TensorShape inshapes[inputCnt];
    TensorShape outshapes[outputCnt];
    shapes.inTensorList  = inshapes;
    shapes.outTensorList = outshapes;
    getTensorShape(runner.get(), &shapes, inputCnt, outputCnt);

    printf("[Model]  %s\n", modelPath.c_str());
    printf("[Input]  %s  size=%d  h=%d  w=%d\n",
           inputTensors[0]->get_name().c_str(),
           shapes.inTensorList[0].size,
           shapes.inTensorList[0].height,
           shapes.inTensorList[0].width);
    printf("[Output] %s  size=%d\n\n",
           outputTensors[0]->get_name().c_str(),
           shapes.outTensorList[0].size);

    runResnet50_singleThread(runner.get(), imageFolder, kinds);

    return 0;
}
