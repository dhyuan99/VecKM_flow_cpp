void local_events_encoding_cuda(
    float* exp_itA,
    float* upd_exp_itA, // cuda
    int*   target_indices,
    float* sum_exp_itA,
    int*   cnt_exp_itA,
    float* exp_ixyA,
    float* At,
    float* txy, float* pred_flow, int num_events, int num_targets,
    int W, int H, int D, int pxl_radius, float t_center, float t_radius, float sqrt_D
);

void neural_network_cuda(
    float* pred_flow, float* exp_itA, float* nn_cache,
    float* w1, float* w2, float* w3,
    int n, int D
);