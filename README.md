# BenchmarkJade

Benchmark JADE2 GPU Performance


# Benchmarks

* `train.py` trains one epoch of dataset.
* `inference.py` performs one epoch of inference of dataset.
* `smooth_inference.py` iterates over the dataset itself and performs multiple-forward passes per input.


# Results

| Cluster | Dataset | Benchmark | Device Name | # Devices | Num Workers | Data Loading Time | Data Processing Time | Forward Time | Backward Time |
| - | - | - | - | -: | -: | -: | -: | -: | -: | 
| TN | ImageNet | Inference | A40 | 4 | 4 | 0.0649 | - | 0.1104 | - |
| TN | ImageNet | Train | A40 | 4 | 4 | 0.0183 | - | 0.1277 | 0.2036 |
| TN | Dummy | Inference | A40 | 4 | 4 | 0.0485 | - | 0.1120 | - |
| TN | Dummy | Train | A40 | 4 | 4 | 0.0466 | - | 0.1306 | 0.2008 |
