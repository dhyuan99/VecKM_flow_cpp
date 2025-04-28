#include "SliceNormalFlowEstimator.h"

#include <iostream>
#include <fstream>
#include <sstream>
#include <string>
#include <cstdlib>
#include <ctime>

float* load_from_txt(const std::string& path, int* size) {
    std::ifstream infile(path);
    if (!infile) {
        std::cerr << "Cannot open file!" << std::endl;
    }

    size_t num_lines;
    infile >> num_lines;
    *size = num_lines;

    float* data = new float[num_lines * 3];

    size_t idx = 0;
    std::string line;
    std::getline(infile, line); // skip the remainder of the first line

    while (std::getline(infile, line) && idx < num_lines * 3) {
        std::istringstream iss(line);
        float x, y, z;
        if (!(iss >> x >> y >> z)) {
            std::cerr << "Invalid line at index " << idx / 3 << std::endl;
            break;
        }
        data[idx++] = x;
        data[idx++] = y;
        data[idx++] = z;
    }

    std::cout << "Loaded " << idx / 3 << " entries." << std::endl;
    std::cout << "First entry: "
              << data[0] << ", " << data[1] << ", " << data[2] << std::endl;
    std::cout << "Last entry: "
              << data[idx - 3] << ", " << data[idx - 2] << ", " << data[idx - 1] << std::endl;

    return data;
}

int* gen_target_indices(int num_events, int num_targets) {
    int* target_indices = new int[num_targets];
    float step_size = (num_events + 0.0) / num_targets;
    for (int i = 0; i < num_targets; ++i) {
        int index = static_cast<int>(i * step_size);
        target_indices[i] = index;
    }
    return target_indices;
}

int main() {
    int dt = 32;
    int k  = 8;

    SliceNormalFlowEstimator estimator(
        "640x480_32ms_C64_k8",
        500000,
        640, 480, 64, k
    );
    int size;
    int num_targets = 10000;
    float* events_txy = load_from_txt("demo_data/events.txt", &size);
    int* target_indices = gen_target_indices(size, num_targets);

    for (int i = 0; i < 100; i++) {   
        clock_t start = clock();
        float* f = estimator.predict_flows(events_txy, size, target_indices, num_targets, events_txy[0], dt / 2000.);    
        clock_t end = clock();
        double elapsed_time = double(end - start) / CLOCKS_PER_SEC;
        std::cout << "compute that many normal flow costs: " << elapsed_time << " seconds" << std::endl;
        delete[] f;
    }
    delete[] events_txy;
}
