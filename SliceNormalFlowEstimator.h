#pragma once
#include <string>

class SliceNormalFlowEstimator {
public:
    SliceNormalFlowEstimator(
        const char* model_path,    // path to the model. The directory contains "At.txt", "Ax.txt", "Ay.txt", "wr0.txt", etc.
        const int max_num_points,  // maximum number of points within each slice. Used for cuda memory allocation.
        const int W,               // width of the image
        const int H,               // height of the image
        const int D,               // dimension of local events encoding
        const int pxl_radius       // pixel radius of neighboring events.
    );
    ~SliceNormalFlowEstimator();
    
    float* local_events_encoding(
        float* events_txy,     int num_events,
        int*   target_indiecs, int num_targets,
        float t_center, float t_radius);
    /** compute the local event encoding for each events.
     * inputs: 
     *     events_txy:     [num_events, 3] contiguous row matrix. 
     *     target_indices: [num_targets] contiguous row vector. It only computes the encoding for the target indices.
     *     t_center, t_radius: the events_t will be normalized by (events_t - t_center) / t_radius.
     * output: events_encoding: [num_events, 2*D] contiguous row matrix.
     * output[:,:D] are the real part, output[:,D:] are the imaginary part.
    **/
    float* predict_flows(
        float* events_txy,     int num_events, 
        int*   target_indices, int num_targets,
        float t_center, float t_radius);
    /** predict the flow for each events.
     * inputs: 
     *     events_txy:     [num_events, 3] contiguous row matrix. 
     *     target_indices: [num_targets] contiguous row vector. It only computes the encoding for the target indices.
     *     t_center, t_radius: the events_t will be normalized by (events_t - t_center) / t_radius.
     * output: pred_flow: [num_events, 2] contiguous row matrix.
     * output[:,0] are the x component, output[:,1] are the y component.
    **/

    int get_D() { return this->D; }

private:
    float* txy;
    float* pred_flow;
    float* exp_itA;
    float* upd_exp_itA;
    int*   target_indices;
    float* sum_exp_itA;
    int*   cnt_exp_itA;
    float* exp_ixyA;
    float* nn_cache;
    float* At;
    float* Ax;
    float* Ay;
    float* w1;
    float* w2;
    float* w3;
    int W;
    int H;
    int D;
    int max_num_points;
    int pxl_radius;
    float sqrt_D;
    void precompute_exp_ixyA(float* exp_ixyA_host, float* Ax, float* Ay);
    void load_txt(const std::string& path, float* data, int size);
    void copyTxtToDevice(const std::string& path, float* device_ptr, int size);
};
