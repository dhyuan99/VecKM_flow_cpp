#include "SliceNormalFlowEstimator.h"
#include "SliceNormalFlowEstimator.cuh"
#include <cuda_runtime.h>
#include <iostream>
#include <cmath>

SliceNormalFlowEstimator::SliceNormalFlowEstimator(
    const char* model_path,
    const int max_num_points,
    const int W,
    const int H,
    const int D,
    const int pxl_radius
) {
    std::cout << "___________ SliceNormalFlowEstimator Initialize ___________" << std::endl;
    std::cout << "Max number of points per slice: " << max_num_points << std::endl;
    std::cout << "Width: "                          << W << std::endl;
    std::cout << "Height: "                         << H << std::endl;
    std::cout << "Dimension: "                      << D << std::endl;
    std::cout << "Pixel radius: "                   << pxl_radius << std::endl;

    std::string fc1_path = std::string(model_path)  + "/w1.txt";
    std::string fc2_path = std::string(model_path)  + "/w2.txt";
    std::string fc3_path = std::string(model_path)  + "/w3.txt";
    std::string At_path = std::string(model_path)   + "/At.txt";
    std::string Ax_path = std::string(model_path)   + "/Ax.txt";
    std::string Ay_path = std::string(model_path)   + "/Ay.txt";

    this -> W = W;
    this -> H = H;
    this -> D = D;
    this -> sqrt_D = sqrt(D);
    this -> pxl_radius = pxl_radius;
    this -> max_num_points = max_num_points;
    
    cudaMalloc((void**)&(this->At),          sizeof(float) * D);
    cudaMalloc((void**)&(this->txy),         sizeof(float) * max_num_points * 3);
    cudaMalloc((void**)&(this->pred_flow),   sizeof(float) * max_num_points * 2);
    cudaMalloc((void**)&(this->target_indices), sizeof(int)* max_num_points);
    cudaMalloc((void**)&(this->exp_itA),     sizeof(float) * max_num_points * 2 * D);
    cudaMalloc((void**)&(this->upd_exp_itA), sizeof(float) * max_num_points * 2 * D);
    cudaMalloc((void**)&(this->nn_cache),    sizeof(float) * max_num_points * 2 * D);
    cudaMalloc((void**)&(this->sum_exp_itA), sizeof(float) * W * H * 2 * D);
    cudaMalloc((void**)&(this->cnt_exp_itA), sizeof(int)   * W * H);
    cudaMalloc((void**)&(this->exp_ixyA),    sizeof(float) * (2*pxl_radius+1) * (2*pxl_radius+1) * 2 * D);
    cudaMalloc((void**)&(this->w1), sizeof(float) * D * D * 4);
    cudaMalloc((void**)&(this->w2), sizeof(float) * D * D * 4);
    cudaMalloc((void**)&(this->w3), sizeof(float) * D * 4);

    size_t free_mem, total_mem;
    cudaMemGetInfo(&free_mem, &total_mem);
    std::cout << "\nCUDA Free: " << free_mem / (1024.0 * 1024) << " MB\n";
    std::cout << "CUDA Total: " << total_mem / (1024.0 * 1024) << " MB\n";
    std::cout << "CUDA Used: " << (total_mem - free_mem) / (1024.0 * 1024) << " MB\n";
    std::cout << "No more cuda memory will be allocated during inference.\n" << std::endl;

    float* Ax = new float[D];
    load_txt(Ax_path, Ax, D);
    float* Ay = new float[D];
    load_txt(Ay_path, Ay, D);
    
    float* exp_ixyA_host = new float[(2*pxl_radius+1) * (2*pxl_radius+1) * 2 * D];
    this -> precompute_exp_ixyA(exp_ixyA_host, Ax, Ay);
    cudaMemcpy(
        this->exp_ixyA, exp_ixyA_host, 
        sizeof(float) * (2*pxl_radius+1) * (2*pxl_radius+1) * 2 * D, 
        cudaMemcpyHostToDevice);
    delete[] exp_ixyA_host;
    delete[] Ax;
    delete[] Ay;

    this->copyTxtToDevice(At_path,  this->At, D);
    this->copyTxtToDevice(fc1_path, this->w1, D * D * 4);
    this->copyTxtToDevice(fc2_path, this->w2, D * D * 4);
    this->copyTxtToDevice(fc3_path, this->w3, D * 4);

    std::cout << "_______________________ Initialization ends _______________________" << std::endl;
}

void SliceNormalFlowEstimator::precompute_exp_ixyA(float* exp_ixyA_host, float* Ax, float* Ay) {
    for (int i = -pxl_radius; i <= pxl_radius; i++) {
        for (int j = -pxl_radius; j <= pxl_radius; j++) {
            for (int k = 0; k < this->D; k++) {
                float xA = (i+0.0) / pxl_radius * Ax[k];
                float yA = (j+0.0) / pxl_radius * Ay[k];
                exp_ixyA_host[
                    (i + pxl_radius) * (2 * pxl_radius + 1) * 2 * D + (j + pxl_radius) * 2 * D + k
                ] = cosf(xA + yA);
                exp_ixyA_host[
                    (i + pxl_radius) * (2 * pxl_radius + 1) * 2 * D + (j + pxl_radius) * 2 * D + k + D
                ] = sinf(xA + yA);
            }
        }
    }
}

void SliceNormalFlowEstimator::load_txt(const std::string& path, float* data, int size) {
    std::cout << "Loading data from: " << path << std::endl;
    // load from txt, where each line is a number
    FILE* file = fopen(path.c_str(), "r");
    if (file == nullptr) {
        std::cerr << "Error opening file: " << path << std::endl;
        return;
    }
    for (int i = 0; i < size; i++) {
        fscanf(file, "%f", &data[i]);
    }
    fclose(file);
    std::cout << "Data loaded successfully." << std::endl;
    std::cout << "first and last element: " << data[0] << " " << data[size-1] << std::endl;
}

void SliceNormalFlowEstimator::copyTxtToDevice(const std::string& path, float* device_ptr, int size) {
    float* host_ptr = new float[size];
    load_txt(path, host_ptr, size);
    cudaMemcpy(device_ptr, host_ptr, sizeof(float) * size, cudaMemcpyHostToDevice);
    delete[] host_ptr;
}

float* SliceNormalFlowEstimator::local_events_encoding(
    float* events_txy,     int num_events,
    int*   target_indiecs, int num_targets,
    float t_center, float t_radius) {
    if (num_events > this->max_num_points) {
        std::cerr << "Number of events per slice exceeds maximum limit. Please set a higher value." << std::endl;
        return nullptr;
    }
    cudaMemcpy(this->txy, events_txy, sizeof(float) * num_events * 3, cudaMemcpyHostToDevice);
    cudaMemcpy(this->target_indices, target_indiecs, sizeof(int) * num_targets, cudaMemcpyHostToDevice);
    local_events_encoding_cuda(
        this->exp_itA, this->upd_exp_itA, this->target_indices,
        this->sum_exp_itA, this->cnt_exp_itA, this->exp_ixyA,
        this->At,
        this->txy, this->pred_flow, num_events, num_targets,
        this->W, this->H, this->D, this->pxl_radius, t_center, t_radius, this->sqrt_D
    );
    float* exp_itA_host = new float[num_targets * 2 * D];
    cudaMemcpy(exp_itA_host, this->upd_exp_itA, sizeof(float) * num_targets * 2 * D, cudaMemcpyDeviceToHost);
    return exp_itA_host;
}

float* SliceNormalFlowEstimator::predict_flows(
    float* events_txy,     int num_events, 
    int*   target_indices, int num_targets,
    float t_center, float t_radius) {
    if (num_events > this->max_num_points) {
        std::cerr << "Number of events exceeds maximum limit." << std::endl;
        return nullptr;
    }
    cudaMemcpy(this->txy, events_txy, sizeof(float) * num_events * 3, cudaMemcpyHostToDevice);
    cudaMemcpy(this->target_indices, target_indices, sizeof(int) * num_targets, cudaMemcpyHostToDevice);
    local_events_encoding_cuda(
        this->exp_itA, this->upd_exp_itA, this->target_indices, 
        this->sum_exp_itA, this->cnt_exp_itA, this->exp_ixyA,
        this->At,
        this->txy, this->pred_flow, num_events, num_targets,
        this->W, this->H, this->D, this->pxl_radius, t_center, t_radius, this->sqrt_D
    );
    neural_network_cuda(
        this->pred_flow, this->upd_exp_itA, this->nn_cache,
        this->w1, this->w2, this->w3,
        num_targets, this->D
    );
    float* pred_flow_host = new float[num_targets * 2];
    cudaMemcpy(pred_flow_host, this->pred_flow, sizeof(float) * num_targets * 2, cudaMemcpyDeviceToHost);
    return pred_flow_host;
}

SliceNormalFlowEstimator::~SliceNormalFlowEstimator() {
    std::cout << this->D << std::endl;
    cudaFree(At);
    cudaFree(txy);
    cudaFree(pred_flow);
    cudaFree(exp_itA);
    cudaFree(upd_exp_itA);
    cudaFree(target_indices);
    cudaFree(sum_exp_itA);
    cudaFree(cnt_exp_itA);
    cudaFree(exp_ixyA);
    cudaFree(nn_cache);
    cudaFree(w1);
    cudaFree(w2);
    cudaFree(w3);
}