<h1 align='center' style="text-align:center; font-weight:bold; font-size:2.0em;letter-spacing:2.0px;"> A Real-Time Event-Based Normal Flow Estimator </h1>

![](./assets/demo.gif)
## Summary
The repository contains C++/CUDA implementation of [VecKM_flow](https://github.com/dhyuan99/VecKM_flow), an event-based normal flow estimator. It contains multiple functions:
1. `SliceNormalFlowEstimator::local_events_encoding`: compute per-event features given a slice of events. The feature is computed from the spatiotemporal neighborhoods of the eventt.
2. `SliceNormalFlowEstimator::predict_flows`: compute per-event normal flow given a slice of events. It first computes per-event feature and then maps the features to predicted normal flows using a two-layer neural network.
3. `_binding.cpp`, `_setup.py` provide a python interface so that the C++/CUDA implementation can be used in python.

## To run the inference in C++:
See an example code at [main.cpp](main.cpp). The folders like `640x480_32ms_C64_k8` contains the model parameters, which stores `At.txt`, `Ax.txt`, `Ay.txt`, `w1.txt`, `w2.txt`, `w3.txt`. You will need to specify the model to use in `main.cpp`.

#### Compile the `.cu` and `.cpp` files into `.so`.
```
nvcc -Xcompiler -fPIC -shared SliceNormalFlowEstimator.cpp SliceNormalFlowEstimator.cu -o libSliceNormalFlowEstimator.so -lcublas
```
This will generate a `libSliceNormalFlowEstimator.so`.

#### Compile the `main.cpp` with the `libSliceNormalFlowEstimator.so` as library.
```
g++ main.cpp -L. -lSliceNormalFlowEstimator -o main
```
This will generate a `main`.

#### Run the inference.
```
LD_LIBRARY_PATH=.
./main
```

#### Example outputs.
```
___________ SliceNormalFlowEstimator Initialize ___________
Max number of points: 500000
Width: 640
Height: 480
Dimension: 64
Pixel radius: 8

CUDA Free: 10011.1 MB
CUDA Total: 10822.9 MB
CUDA Used: 811.812 MB
No more cuda memory will be allocated during inference.

Loading data from: 640x480_32ms_C64_k8/Ax.txt
Data loaded successfully.
first and last element: -1.46691 1.06858
Loading data from: 640x480_32ms_C64_k8/Ay.txt
Data loaded successfully.
first and last element: 0.87208 1.66946
Loading data from: 640x480_32ms_C64_k8/At.txt
Data loaded successfully.
first and last element: 2.73328 2.51201
Loading data from: 640x480_32ms_C64_k8/w1.txt
Data loaded successfully.
first and last element: -0.147332 -0.0802209
Loading data from: 640x480_32ms_C64_k8/w2.txt
Data loaded successfully.
first and last element: -0.0154236 -0.104303
Loading data from: 640x480_32ms_C64_k8/w3.txt
Data loaded successfully.
first and last element: -0.052781 -0.362976
_______________________ Initialization ends _______________________
Loaded 122219 entries.
First entry: 0.0400636, 0, 68
Last entry: 0.0698538, 639, 212
compute that many normal flow costs: 0.037906 seconds
compute that many normal flow costs: 0.030989 seconds
compute that many normal flow costs: 0.030952 seconds
compute that many normal flow costs: 0.030973 seconds
compute that many normal flow costs: 0.030932 seconds
compute that many normal flow costs: 0.03097 seconds
compute that many normal flow costs: 0.03103 seconds
compute that many normal flow costs: 0.030802 seconds
compute that many normal flow costs: 0.030755 seconds
compute that many normal flow costs: 0.030748 seconds
64
```

## To run the inference in Python:
#### install pybind11
```
pip install pybind11
```

#### compile a python-loadable library:
```
python _setup.py build_ext --inplace
```
This will generate a `VecKM_flow.cpython-311-x86_64-linux-gnu.so`.

#### Run the inference:
```
python main.py
```
It will load the data from [./demo_data](demo_data) and produce a flow prediction video named `demo.mp4`.

## Potential CUDA Driver Issues
This is the CUDA driver I used. If it raises `Bus Error core dump`, it is most likely the CUDA driver needs to be updated.
```
+-----------------------------------------------------------------------------------------+
| NVIDIA-SMI 570.133.07             Driver Version: 570.133.07     CUDA Version: 12.8     |
|-----------------------------------------+------------------------+----------------------+
| GPU  Name                 Persistence-M | Bus-Id          Disp.A | Volatile Uncorr. ECC |
| Fan  Temp   Perf          Pwr:Usage/Cap |           Memory-Usage | GPU-Util  Compute M. |
|                                         |                        |               MIG M. |
|=========================================+========================+======================|
|   0  NVIDIA GeForce RTX 2080 Ti     Off |   00000000:3D:00.0 Off |                  N/A |
| 32%   25C    P8              1W /  250W |       1MiB /  11264MiB |      0%      Default |
|                                         |                        |                  N/A |
+-----------------------------------------+------------------------+----------------------+
```

## Citations
```
```

```
@article{yuan2024learning,
  title={Learning Normal Flow Directly From Event Neighborhoods},
  author={Yuan, Dehao and Burner, Levi and Wu, Jiayi and Liu, Minghui and Chen, Jingxi and Aloimonos, Yiannis and Ferm{\"u}ller, Cornelia},
  journal={arXiv preprint arXiv:2412.11284},
  year={2024}
}
```