#include "SliceNormalFlowEstimator.cuh"
#include <cuda_runtime.h>
#include <math.h>
#include <cublas_v2.h>

__global__ void normalize_time(float* txy, int num_events, float t_center, float t_radius) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < num_events) {
        txy[idx*3] = (txy[idx*3] - t_center) / t_radius;
    }
}   

__global__ void init_sum_exp_itA(float* sum_exp_itA, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        sum_exp_itA[idx] = 0.0f;
    }
}

__global__ void init_cnt_exp_itA(int* cnt_exp_itA, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        cnt_exp_itA[idx] = 0;
    }
}

__global__ void compute_exp_itA(
    float* exp_itA, float* txy, float* At, int num_events, int D) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int n_idx = idx / D;
    int d_idx = idx % D;
    if (idx < num_events * D) {
        float tmp = txy[n_idx * 3] * At[d_idx];
        exp_itA[n_idx * (2 * D) + d_idx    ] = cosf(tmp);
        exp_itA[n_idx * (2 * D) + d_idx + D] = sinf(tmp);
    }
}

__global__ void compute_sum_exp_itA(
    float* sum_exp_itA, int* cnt_exp_itA, float* exp_itA, float* txy, 
    int num_events, int D, int W, int H
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int n_idx = idx / D;
    int d_idx = idx % D;
    if (idx < num_events * D) {
        int x = __float2int_rn(txy[n_idx * 3 + 1]);
        int y = __float2int_rn(txy[n_idx * 3 + 2]);
        if (x >= 0 && x < W && y >= 0 && y < H) {
            atomicAdd(
                &sum_exp_itA[x * (H * 2 * D) + y * (2 * D) + d_idx], 
                exp_itA[n_idx * (2 * D) + d_idx]);
            atomicAdd(
                &sum_exp_itA[x * (H * 2 * D) + y * (2 * D) + d_idx + D], 
                exp_itA[n_idx * (2 * D) + d_idx + D]);
            if (d_idx == 0) {
                atomicAdd(
                    &cnt_exp_itA[x * H + y], 1);
            }
        }
    }
}

__global__ void update_exp_itA(
    float* exp_itA, float* upd_exp_itA, int* target_indices,
    float* sum_exp_itA, int* cnt_exp_itA, float* txy, float* exp_ixyA, 
    int num_events, int num_targets, int D, int W, int H, int pxl_radius, float sqrt_D
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int t_idx = idx / D;
    int d_idx = idx % D;

    int n_idx = target_indices[t_idx];
    if (n_idx < 0 || n_idx >= num_events) {
        return;
    }
    if (idx < num_targets * D) {
        int x = __float2int_rn(txy[n_idx * 3 + 1]);
        int y = __float2int_rn(txy[n_idx * 3 + 2]);
        float cos_txyA = 0.0f;
        float sin_txyA = 0.0f;
        int   count = 0;
        for (int dx = -pxl_radius; dx <= pxl_radius; dx++) {
            for (int dy = -pxl_radius; dy <= pxl_radius; dy++) {
                int x_idx = x + dx;
                int y_idx = y + dy;
                if (x_idx >= 0 && x_idx < W && y_idx >= 0 && y_idx < H) {
                    float cos_xyA = exp_ixyA[
                        (dx+pxl_radius) * (2*pxl_radius+1) * 2 * D + (dy+pxl_radius) * (2 * D) + d_idx
                    ];
                    float sin_xyA = exp_ixyA[
                        (dx+pxl_radius) * (2*pxl_radius+1) * 2 * D + (dy+pxl_radius) * (2 * D) + d_idx + D
                    ];
                    float cos_tA = sum_exp_itA[x_idx * (H * 2 * D) + y_idx * (2 * D) + d_idx];
                    float sin_tA = sum_exp_itA[x_idx * (H * 2 * D) + y_idx * (2 * D) + d_idx + D];
                    cos_txyA += cos_tA * cos_xyA - sin_tA * sin_xyA;
                    sin_txyA += sin_tA * cos_xyA + cos_tA * sin_xyA;
                    count += cnt_exp_itA[x_idx * H + y_idx];
                }
            }
        }
        float cur_cos_tA = exp_itA[n_idx * (2 * D) + d_idx];
        float cur_sin_tA = exp_itA[n_idx * (2 * D) + d_idx + D];
        float upd_cos_tA = (cur_cos_tA * cos_txyA + cur_sin_tA * sin_txyA) / (cur_cos_tA * cur_cos_tA + cur_sin_tA * cur_sin_tA);
        float upd_sin_tA = (sin_txyA * cur_cos_tA - cos_txyA * cur_sin_tA) / (cur_cos_tA * cur_cos_tA + cur_sin_tA * cur_sin_tA);
        upd_exp_itA[t_idx * (2 * D) + d_idx]     = upd_cos_tA / count * sqrt_D;  
        upd_exp_itA[t_idx * (2 * D) + d_idx + D] = upd_sin_tA / count * sqrt_D;
    }
}

__global__ void relu_inplace(float* x, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        x[idx] = fmaxf(x[idx], 0.0f);
    }
}

void local_events_encoding_cuda(
    float* exp_itA,     // cuda
    float* upd_exp_itA, // cuda
    int*   target_indices,
    float* sum_exp_itA, // cuda
    int*   cnt_exp_itA, // cuda
    float* exp_ixyA,    // cuda
    float* At,
    float* txy, float* pred_flow, int num_events, int num_targets,
    int W, int H, int D, int pxl_radius, float t_center, float t_radius, float sqrt_D
) {
    // Launch kernel
    int threads = 256;
    int blocks_n    = (num_events   + threads - 1) / threads;
    int blocks_nd   = (num_events*D + threads - 1) / threads;
    int blocks_td   = (num_targets*D + threads - 1) / threads;
    int blocks_WH   = (W*H          + threads - 1) / threads;
    int blocks_2WHD = (2*W*H*D      + threads - 1) / threads;
    
    // Normalize time
    normalize_time       <<<blocks_n,    threads>>> (txy, num_events, t_center, t_radius);
    init_sum_exp_itA     <<<blocks_2WHD, threads>>> (sum_exp_itA, 2*W*H*D);
    init_cnt_exp_itA     <<<blocks_WH,   threads>>> (cnt_exp_itA, W*H);
    compute_exp_itA      <<<blocks_nd,   threads>>> (exp_itA, txy, At, num_events, D);
    compute_sum_exp_itA  <<<blocks_nd,   threads>>> (sum_exp_itA, cnt_exp_itA, exp_itA, txy, num_events, D, W, H);
    update_exp_itA       <<<blocks_td,   threads>>> (
        exp_itA, upd_exp_itA, target_indices, 
        sum_exp_itA, cnt_exp_itA, txy, exp_ixyA, 
        num_events, num_targets, D, W, H, pxl_radius, sqrt_D);
}

void neural_network_cuda(
    float* pred_flow, float* exp_itA, float* nn_cache,
    float* w1, float* w2, float* w3,
    int n, int D
) {
    float alpha = 1.0f;
    float beta = 0.0f;
    int threads = 256;
    int blocks_2nd   = (2*n*D + threads - 1) / threads;
    int D2 = 2*D;
    
    cublasHandle_t handle;
    cublasCreate(&handle);
    
    cublasSgemm(handle,
        CUBLAS_OP_N,
        CUBLAS_OP_N,
        D2,
        n,
        D2,
        &alpha,
        w1, D2,
        exp_itA, D2,
        &beta,
        nn_cache, D2);

    relu_inplace <<<blocks_2nd, threads>>> (nn_cache, D2*n);

    cublasSgemm(handle,
        CUBLAS_OP_N,
        CUBLAS_OP_N,
        D2,
        n,
        D2,
        &alpha,
        w2, D2,
        nn_cache, D2,
        &beta,
        exp_itA, D2);

    relu_inplace <<<blocks_2nd, threads>>> (exp_itA, D2*n);

    cublasSgemm(handle,
        CUBLAS_OP_N,
        CUBLAS_OP_N,
        2,
        n,
        D2,
        &alpha,
        w3, 2,
        exp_itA, D2,
        &beta,
        pred_flow, 2);
    
    cublasDestroy(handle);
    
}