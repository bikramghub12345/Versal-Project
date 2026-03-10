/*
 * main_multiThread.cc
 *
 * Multi-threaded ResNet50 inference over all images in a folder.
 *
 * Threading model (standard Vitis-AI pattern):
 *   - Each worker thread owns its own vart::Runner (thread-safe by design)
 *   - A shared input queue feeds image paths to all workers
 *   - Each worker: imread → preprocess → DPU → softmax → push result
 *   - Main thread collects results and prints the report
 *
 * Usage: ./resnet50_mt <model.xmodel> <image_folder> [num_threads]
 *        num_threads defaults to 2 (ZCU104 DPUCZDX8G has 2 DPU cores)
 *
 * Build:
 *   g++ -std=c++17 -O2 -o resnet50_mt src/main_multiThread.cc \
 *       ../common/common.cpp -I./src -I../common \
 *       -I/usr/include/opencv4 -I/usr/include/vitis_ai \
 *       $(pkg-config --cflags --libs opencv4) \
 *       -lvart-runner -lxir -lglog -lpthread
 */

#include <assert.h>
#include <dirent.h>
#include <stdio.h>
#include <stdlib.h>
#include <sys/stat.h>
#include <unistd.h>

#include <algorithm>
#include <atomic>
#include <chrono>
#include <condition_variable>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <mutex>
#include <queue>
#include <string>
#include <thread>
#include <vector>

#include "common.h"
#include <opencv2/opencv.hpp>

using namespace std;
using namespace cv;
using namespace std::chrono;

// ─────────────────────────────────────────────────────────────────────────────
// Globals set once at startup
// ─────────────────────────────────────────────────────────────────────────────
GraphInfo shapes;
const string wordsPath = "./";

// ─────────────────────────────────────────────────────────────────────────────
// Result record — one per image, filled by worker threads
// ─────────────────────────────────────────────────────────────────────────────
struct ImageResult {
    string   filename;
    int      top1_class   = -1;
    float    top1_prob    = 0.f;
    int      top5[5]      = {};
    double   time_ms      = 0.0;   // preprocess + DPU + postprocess
    int      thread_id    = -1;
};

// ─────────────────────────────────────────────────────────────────────────────
// Thread-safe input queue
// ─────────────────────────────────────────────────────────────────────────────
struct ImageQueue {
    mutex              mtx;
    condition_variable cv;
    queue<string>      q;
    bool               done = false;

    void push(const string& s) {
        lock_guard<mutex> lk(mtx);
        q.push(s);
        cv.notify_one();
    }

    // Returns false when queue is empty AND done==true
    bool pop(string& out) {
        unique_lock<mutex> lk(mtx);
        cv.wait(lk, [&]{ return !q.empty() || done; });
        if (q.empty()) return false;
        out = q.front(); q.pop();
        return true;
    }

    void set_done() {
        lock_guard<mutex> lk(mtx);
        done = true;
        cv.notify_all();
    }
};

// ─────────────────────────────────────────────────────────────────────────────
// Thread-safe output queue
// ─────────────────────────────────────────────────────────────────────────────
struct ResultQueue {
    mutex              mtx;
    vector<ImageResult> results;

    void push(ImageResult&& r) {
        lock_guard<mutex> lk(mtx);
        results.push_back(move(r));
    }
};

// ─────────────────────────────────────────────────────────────────────────────
// Helper functions
// ─────────────────────────────────────────────────────────────────────────────
void ListImages(const string& path, vector<string>& images) {
    images.clear();
    struct stat s;
    lstat(path.c_str(), &s);
    if (!S_ISDIR(s.st_mode)) {
        fprintf(stderr, "Error: %s is not a valid directory!\n", path.c_str());
        exit(1);
    }
    DIR* dir = opendir(path.c_str());
    if (!dir) { fprintf(stderr, "Error: Cannot open %s\n", path.c_str()); exit(1); }
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
    sort(images.begin(), images.end());
}

void LoadWords(const string& path, vector<string>& kinds) {
    kinds.clear();
    ifstream f(path);
    if (f.fail()) { fprintf(stderr, "Error: Cannot open %s\n", path.c_str()); exit(1); }
    string line;
    while (getline(f, line)) kinds.push_back(line);
}

void CPUCalcSoftmax(const int8_t* data, size_t size, float* result, float scale) {
    double sum = 0.0;
    for (size_t i = 0; i < size; i++) { result[i] = exp((float)data[i] * scale); sum += result[i]; }
    for (size_t i = 0; i < size; i++) result[i] /= sum;
}

// Returns top-k class indices sorted by descending probability
vector<int> TopKIdx(const float* d, int size, int k) {
    vector<pair<float,int>> v(size);
    for (int i = 0; i < size; i++) v[i] = {d[i], i};
    sort(v.begin(), v.end(), greater<pair<float,int>>());
    vector<int> idx(min(k,size));
    for (int i = 0; i < (int)idx.size(); i++) idx[i] = v[i].second;
    return idx;
}

// ─────────────────────────────────────────────────────────────────────────────
// Worker thread function
// Each thread creates its own runner → independent DPU context
// ─────────────────────────────────────────────────────────────────────────────
void worker(int thread_id,
            const xir::Subgraph* subgraph,
            const string& imageFolder,
            ImageQueue& inQ,
            ResultQueue& outQ,
            atomic<int>& active_threads)
{
    // Each thread gets its own runner — this is the key to Vitis-AI multithreading.
    // Runners are independent: they don't share state and can run concurrently.
    auto runner = vart::Runner::create_runner(subgraph, "run");

    auto inputTensors  = runner->get_input_tensors();
    auto outputTensors = runner->get_output_tensors();

    float input_scale  = get_input_scale(inputTensors[0]);
    float output_scale = get_output_scale(outputTensors[0]);

    int outSize  = shapes.outTensorList[0].size;
    int inSize   = shapes.inTensorList[0].size;
    int inHeight = shapes.inTensorList[0].height;
    int inWidth  = shapes.inTensorList[0].width;

    const float mean[3] = {104.f, 107.f, 123.f};

    // Per-thread buffers (no sharing needed)
    vector<int8_t> imageInputs(inSize);
    vector<int8_t> FCResult(outSize);
    vector<float>  softmax(outSize);

    string imgName;
    while (inQ.pop(imgName)) {
        auto t0 = steady_clock::now();

        // ── imread ───────────────────────────────────────────────────────
        Mat image = imread(imageFolder + "/" + imgName);
        if (image.empty()) {
            fprintf(stderr, "[Thread %d] Cannot read %s, skipping\n",
                    thread_id, imgName.c_str());
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

        // ── Tensor buffers ───────────────────────────────────────────────
        auto in_dims  = inputTensors[0]->get_shape();  in_dims[0]  = 1;
        auto out_dims = outputTensors[0]->get_shape(); out_dims[0] = 1;

        vector<shared_ptr<xir::Tensor>> bt;
        bt.push_back(shared_ptr<xir::Tensor>(
            xir::Tensor::create(inputTensors[0]->get_name(), in_dims,
                                xir::DataType{xir::DataType::XINT, 8u})));
        bt.push_back(shared_ptr<xir::Tensor>(
            xir::Tensor::create(outputTensors[0]->get_name(), out_dims,
                                xir::DataType{xir::DataType::XINT, 8u})));

        vector<unique_ptr<vart::TensorBuffer>> inputs, outputs;
        inputs.push_back(make_unique<CpuFlatTensorBuffer>(imageInputs.data(), bt[0].get()));
        outputs.push_back(make_unique<CpuFlatTensorBuffer>(FCResult.data(),   bt[1].get()));

        vector<vart::TensorBuffer*> inputsPtr  = {inputs[0].get()};
        vector<vart::TensorBuffer*> outputsPtr = {outputs[0].get()};

        // ── DPU inference ────────────────────────────────────────────────
        auto job_id = runner->execute_async(inputsPtr, outputsPtr);
        runner->wait(job_id.first, -1);

        // ── Postprocess ──────────────────────────────────────────────────
        CPUCalcSoftmax(FCResult.data(), outSize, softmax.data(), output_scale);
        auto tk = TopKIdx(softmax.data(), outSize, 5);

        double img_ms = duration<double,milli>(steady_clock::now() - t0).count();

        ImageResult R;
        R.filename  = imgName;
        R.top1_class= tk[0];
        R.top1_prob = softmax[tk[0]];
        for (int i = 0; i < 5 && i < (int)tk.size(); i++) R.top5[i] = tk[i];
        R.time_ms   = img_ms;
        R.thread_id = thread_id;

        outQ.push(move(R));
    }

    active_threads--;
}

// ─────────────────────────────────────────────────────────────────────────────
// Entry point
// ─────────────────────────────────────────────────────────────────────────────
int main(int argc, char* argv[]) {
    if (argc < 3 || argc > 4) {
        printf("Usage: %s <model.xmodel> <image_folder> [num_threads]\n", argv[0]);
        return 1;
    }

    const string modelPath   = argv[1];
    const string imageFolder = argv[2];

    // num_threads: from arg, or ask user interactively
    int numThreads = 0;
    if (argc == 4) {
        numThreads = atoi(argv[3]);
    } else {
        printf("Enter number of threads [1-8, recommended 2 for ZCU104]: ");
        fflush(stdout);
        scanf("%d", &numThreads);
    }
    if (numThreads < 1) numThreads = 1;
    if (numThreads > 8) numThreads = 8;

    // Load labels
    vector<string> kinds;
    LoadWords(wordsPath + "words.txt", kinds);
    if (kinds.empty()) {
        fprintf(stderr, "Error: words.txt not found\n"); return 1;
    }

    // List images
    vector<string> images;
    ListImages(imageFolder, images);
    if (images.empty()) {
        fprintf(stderr, "Error: No images found in %s\n", imageFolder.c_str()); return 1;
    }

    // Deserialize model — shared across all threads (read-only)
    auto graph    = xir::Graph::deserialize(modelPath);
    auto subgraph = get_dpu_subgraph(graph.get());
    if (subgraph.size() != 1u) {
        fprintf(stderr, "Error: Expected 1 DPU subgraph, got %zu\n", subgraph.size());
        return 1;
    }

    // Populate shapes. Use static arrays so shapes.inTensorList/outTensorList
    // pointers remain valid for the entire program lifetime (VLAs on the stack
    // inside a block would be destroyed when the block exits).
    static TensorShape inshapes[8];   // 8 is more than enough for any model
    static TensorShape outshapes[8];
    shapes.inTensorList  = inshapes;
    shapes.outTensorList = outshapes;
    {
        auto shape_runner = vart::Runner::create_runner(subgraph[0], "run");
        auto inT  = shape_runner->get_input_tensors();
        auto outT = shape_runner->get_output_tensors();
        getTensorShape(shape_runner.get(), &shapes,
                       (int)inT.size(), (int)outT.size());
    }
    if (shapes.inTensorList[0].size == 0 || shapes.outTensorList[0].size == 0) {
        fprintf(stderr, "Error: getTensorShape returned zero sizes\n");
        return 1;
    }

    // Print config
    printf("════════════════════════════════════════════\n");
    printf("  ResNet50 Multi-Thread Inference\n");
    printf("════════════════════════════════════════════\n");
    printf("  Model       : %s\n", modelPath.c_str());
    printf("  Image folder: %s\n", imageFolder.c_str());
    printf("  Images      : %zu\n", images.size());
    printf("  Threads     : %d\n", numThreads);
    printf("  Input size  : %d  (%dx%d)\n",
           shapes.inTensorList[0].size,
           shapes.inTensorList[0].height,
           shapes.inTensorList[0].width);
    printf("  Output size : %d\n", shapes.outTensorList[0].size);
    printf("════════════════════════════════════════════\n\n");

    // Fill input queue
    ImageQueue  inQ;
    ResultQueue outQ;
    for (auto& img : images) inQ.push(img);
    inQ.set_done();  // no more images will be added

    // Launch worker threads
    atomic<int> active(numThreads);
    vector<thread> threads;
    auto wall_start = steady_clock::now();

    for (int t = 0; t < numThreads; t++) {
        threads.emplace_back(worker, t, subgraph[0],
                             ref(imageFolder), ref(inQ), ref(outQ), ref(active));
    }

    // Wait for all threads
    for (auto& t : threads) t.join();

    double wall_ms = duration<double,milli>(steady_clock::now() - wall_start).count();

    // ── Sort results by filename for consistent output ────────────────────
    auto& results = outQ.results;
    sort(results.begin(), results.end(),
         [](const ImageResult& a, const ImageResult& b){
             return a.filename < b.filename; });

    // ── Print per-image results ───────────────────────────────────────────
    for (size_t i = 0; i < results.size(); i++) {
        auto& R = results[i];
        printf("[%4zu / %4zu]  %-30s  time=%6.2f ms  thread=%d\n",
               i+1, results.size(), R.filename.c_str(), R.time_ms, R.thread_id);
        printf("  top[0]  prob=%.6f  class=%d  %s\n",
               R.top1_prob, R.top1_class,
               R.top1_class < (int)kinds.size() ? kinds[R.top1_class].c_str() : "?");
        for (int k = 1; k < 5; k++) {
            int cls = R.top5[k];
            printf("  top[%d]  prob=%.6f  class=%d  %s\n",
                   k, 0.f, cls,  // prob for top2-5 not stored to keep struct small
                   cls < (int)kinds.size() ? kinds[cls].c_str() : "?");
        }
        printf("\n");
    }

    // ── Summary ──────────────────────────────────────────────────────────
    size_t N = results.size();
    if (N > 0) {
        double sum_ms = 0.0, min_ms = 1e9, max_ms = 0.0;
        vector<double> per_thread(numThreads, 0.0);
        vector<int>    per_thread_cnt(numThreads, 0);

        for (auto& R : results) {
            sum_ms += R.time_ms;
            min_ms  = min(min_ms, R.time_ms);
            max_ms  = max(max_ms, R.time_ms);
            if (R.thread_id >= 0 && R.thread_id < numThreads) {
                per_thread[R.thread_id]     += R.time_ms;
                per_thread_cnt[R.thread_id] += 1;
            }
        }
        double avg_ms  = sum_ms / N;
        double fps_wall= N * 1000.0 / wall_ms;
        // Theoretical max FPS if threads were perfectly balanced
        double fps_theo= numThreads * 1000.0 / avg_ms;

        printf("════════════════════════════════════════════\n");
        printf("  TIMING SUMMARY  (%zu images, %d threads)\n", N, numThreads);
        printf("════════════════════════════════════════════\n");
        printf("  Total wall time  : %8.2f ms  (%.3f s)\n", wall_ms, wall_ms/1000.0);
        printf("  Sum of img times : %8.2f ms\n", sum_ms);
        printf("  Avg time/image   : %8.2f ms\n", avg_ms);
        printf("  Min time/image   : %8.2f ms\n", min_ms);
        printf("  Max time/image   : %8.2f ms\n", max_ms);
        printf("────────────────────────────────────────────\n");
        printf("  FPS (wall clock) : %8.2f  <- actual throughput\n", fps_wall);
        printf("  FPS (theoretical): %8.2f  <- %d x (1000/%.2fms)\n",
               fps_theo, numThreads, avg_ms);
        printf("────────────────────────────────────────────\n");
        printf("  Per-thread breakdown:\n");
        for (int t = 0; t < numThreads; t++) {
            int cnt = per_thread_cnt[t];
            double tavg = cnt > 0 ? per_thread[t] / cnt : 0.0;
            printf("    Thread %d : %3d images  avg=%.2f ms\n", t, cnt, tavg);
        }
        printf("════════════════════════════════════════════\n");
    }

    printf("\n[Done] Processed %zu / %zu images  |  %d threads\n",
           N, images.size(), numThreads);
    return 0;
}
