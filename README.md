# BenchmarkJade

Benchmark JADE2 GPU Performance


# Benchmarks

* `train.py` trains one epoch of dataset.
* `inference.py` performs one epoch of inference of dataset.
* `smooth_inference.py` iterates over the dataset itself and performs multiple-forward passes per input.



## Results

### Resnet 50 (fp32)
| Cluster | Dataset | Benchmark | Device Name | # Devices | Num Workers | Data Loading Time | Data Processing Time | Forward Time | Backward Time |
| - | - | - | - | -: | -: | -: | -: | -: | -: |
| TN | ImageNet | Inference | V100 | 1 | 4 | 0.1022 | - | 0.1128 | - |
| TN | ImageNet | Train | V100 | 1 | 4 | 0.0100 | - | 0.1233 | 0.2378 |
| TN | ImageNet | Smooth Inference | V100 | 1 | 4 | 0.1009 | 0.0009 | 0.0082 | - |
| TN | Dummy | Inference | V100 | 1 | 4 | 0.0317 | - | 0.1116 | - |
| TN | Dummy | Train | V100 | 1 | 4 | 0.0294 | - | 0.1226 | 0.2377 |
| TN | Dummy | Smooth Inference | V100 | 1 | 4 | 0.1014 | 0.0009 | 0.0079 | - |
| TN | ImageNet | Inference | A40 | 1 | 4 | 0.0403 | - | 0.0962 | - |
| TN | ImageNet | Train | A40 | 1 | 4 | 0.0081 | - | 0.1102 | 0.1961 |
| TN | ImageNet | Smooth Inference | A40 | 1 | 4 | 0.0882 | 0.0013 | 0.0048 | - |
| TN | Dummy | Inference | A40 | 1 | 4 | 0.0267 | - | 0.0955 | - |
| TN | Dummy | Train | A40 | 1 | 4 | 0.0268 | - | 0.1096 | 0.1957 |
| TN | Dummy | Smooth Inference | A40 | 1 | 4 | 0.0880 | 0.0012 | 0.0049 | - |
| TN | ImageNet | Inference | A40 | 8 | 4 | 0.0913 | - | 0.1321 | - |
| TN | ImageNet | Train | A40 | 8 | 4 | 0.0406 | - | 0.1472 | 0.2115 |
| TN | ImageNet | Smooth Inference | A40 | 8 | 4 | 0.2392 | 0.0019 | 0.0068 | - |
| TN | Dummy | Inference | A40 | 8 | 4 | 0.0539 | - | 0.1251 | - |
| TN | Dummy | Train | A40 | 8 | 4 | 0.0601 | - | 0.1422 | 0.2109 |
| TN | Dummy | Smooth Inference | A40 | 8 | 4 | 0.3265 | 0.0020 | 0.0074 | - |
| JADE2 | ImageNet | Inference | V100 | 4 | 4 | 5.8355 | - | 1.1637 | - |
| JADE2 | ImageNet | Train | V100 | 4 | 4 | 5.2553 | - | 1.3720 | 0.8037 |
| JADE2 | ImageNet | Smooth Inference | V100 | 4 | 4 | 0.2201 | 0.0503 | 0.0400 | - |
| JADE2 | Dummy | Inference | V100 | 4 | 4 | 3.3424 | - | 0.9132 | - |
| JADE2 | Dummy | Train | V100 | 4 | 4 | 2.8620 | - | 1.0636 | 0.6111 |
| JADE2 | Dummy | Smooth Inference | V100 | 4 | 4 | 0.1844 | 0.0468 | 0.559 | - |
| JADE2 | ImageNet | Inference | V100 | 1 | 4 | 1.0153 | - | 0.1510 | - |
| JADE2 | ImageNet | Train | V100 | 1 | 4 | 0.6594 | - | 0.1774 | 0.3166 |
| JADE2 | ImageNet | Smooth Inference | V100 | 1 | 4 | 0.1281 | 0.0023 | 0.0121 | - |
| JADE2 | Dummy | Inference | V100 | 1 | 4 | 0.4596 | - | 0.1871 | - |
| JADE2 | Dummy | Train | V100 | 1 | 4 | 0.2058 | - | 0.2137 | 0.3367 |
| JADE2 | Dummy | Smooth Inference | V100 | 1 | 4 | 0.1302 | 0.0026 | 0.0121 | - |


### Expected Speed
[LambaLabs Benchmark](https://lambdalabs.com/gpu-benchmarks)

***FP32***

8 x V100 0.927x as fast as 8 x A40

1 x V100 0.943x as fast as 1 x A40

JADE V100 0.67x as fast as V100

8 x Jade V100 0.621 x as fast as 8 A40

### Comparison

Inference: JADE2 is 9.38 times slower than a A40. This is around 5.8 x slower than it should be.
Train
