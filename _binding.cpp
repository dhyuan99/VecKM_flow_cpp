#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include "SliceNormalFlowEstimator.h"

namespace py = pybind11;

PYBIND11_MODULE(VecKM_flow, m) {
    py::class_<SliceNormalFlowEstimator>(m, "SliceNormalFlowEstimator")
        .def(py::init<const char*, int, int, int, int, int>())
        .def("get_D", &SliceNormalFlowEstimator::get_D)

        .def("local_events_encoding", [](SliceNormalFlowEstimator& self,
                                         py::array_t<float, py::array::c_style | py::array::forcecast> events_txy,
                                         int num_events,
                                         py::array_t<int, py::array::c_style | py::array::forcecast> target_indices,
                                         int num_targets,
                                         float t_center,
                                         float t_radius) {
            if (events_txy.ndim() != 2 || events_txy.shape(1) != 3)
                throw std::runtime_error("events_txy must be of shape (N, 3)");

            float* input_ptr = static_cast<float*>(events_txy.mutable_data());
            int* target_indices_ptr = static_cast<int*>(target_indices.mutable_data());
            float* output_ptr = self.local_events_encoding(input_ptr, num_events, target_indices_ptr, num_targets, t_center, t_radius);
            int D = self.get_D();
            py::capsule free_when_done_capsule(output_ptr, [](void* p) {
                float* fp = static_cast<float*>(p);
                delete[] fp;
            });
            return py::array_t<float>({num_targets, 2 * D}, output_ptr, free_when_done_capsule);
        })

        .def("predict_flows", [](SliceNormalFlowEstimator& self,
                                 py::array_t<float, py::array::c_style | py::array::forcecast> events_txy,
                                 int num_events,
                                 py::array_t<int, py::array::c_style | py::array::forcecast> target_indices,
                                 int num_targets,
                                 float t_center,
                                 float t_radius) {
            if (events_txy.ndim() != 2 || events_txy.shape(1) != 3)
                throw std::runtime_error("events_txy must be of shape (N, 3)");

            float* input_ptr = static_cast<float*>(events_txy.mutable_data());
            int* target_indices_ptr = static_cast<int*>(target_indices.mutable_data());
            float* output_ptr = self.predict_flows(input_ptr, num_events, target_indices_ptr, num_targets, t_center, t_radius);
            py::capsule free_when_done_capsule(output_ptr, [](void* p) {
                float* fp = static_cast<float*>(p);
                delete[] fp;
            });
            return py::array_t<float>({num_targets, 2}, output_ptr, free_when_done_capsule);
        });
}
