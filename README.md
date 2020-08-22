# PCNN

分段卷积神经网络，关系抽取，代码基于Tensorflow2.2实现。



## PCNN实现说明

一些改进：

- 多实例学习warm up，达到warm up steps后才使用多实例学习；
- Adam学习率warm up；
- 基于TextCNN思想，使用不同窗口卷积核对文本卷积，再对每个卷积核的输出按两个实体位置分段，输出三个值；



## 运行项目

**1.克隆项目**

```bash
!git clone https://github.com/dwdb/pcnn.git /content/pcnn
```

**2.切换工作目录**

```bash
!cd /content/pcnn
```

**3.运行项目**

```bash
!python pcnn.py
```

以下是在**Google Colab**中的运行日志：

```
2020-06-07 16:01:02.653840: I tensorflow/stream_executor/platform/default/dso_loader.cc:44] Successfully opened dynamic library libcudart.so.10.1
labels:  {'NA': 0, '/location/country/administrative_divisions': 1, '/location/administrative_division/country': 2, '/location/country/capital': 3, '/people/person/nationality': 4, '/people/person/place_lived': 5, '/location/neighborhood/neighborhood_of': 6, '/business/person/company': 7, '/location/location/contains': 8, '/people/ethnicity/includes_groups': 9}
2020-06-07 16:01:05.560123: I tensorflow/stream_executor/platform/default/dso_loader.cc:44] Successfully opened dynamic library libcuda.so.1
2020-06-07 16:01:05.604370: I tensorflow/stream_executor/cuda/cuda_gpu_executor.cc:981] successful NUMA node read from SysFS had negative value (-1), but there must be at least one NUMA node, so returning NUMA node zero
2020-06-07 16:01:05.604958: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1561] Found device 0 with properties: 
pciBusID: 0000:00:04.0 name: Tesla P100-PCIE-16GB computeCapability: 6.0
coreClock: 1.3285GHz coreCount: 56 deviceMemorySize: 15.90GiB deviceMemoryBandwidth: 681.88GiB/s
2020-06-07 16:01:05.604990: I tensorflow/stream_executor/platform/default/dso_loader.cc:44] Successfully opened dynamic library libcudart.so.10.1
2020-06-07 16:01:05.846374: I tensorflow/stream_executor/platform/default/dso_loader.cc:44] Successfully opened dynamic library libcublas.so.10
2020-06-07 16:01:05.976986: I tensorflow/stream_executor/platform/default/dso_loader.cc:44] Successfully opened dynamic library libcufft.so.10
2020-06-07 16:01:05.997908: I tensorflow/stream_executor/platform/default/dso_loader.cc:44] Successfully opened dynamic library libcurand.so.10
2020-06-07 16:01:06.264169: I tensorflow/stream_executor/platform/default/dso_loader.cc:44] Successfully opened dynamic library libcusolver.so.10
2020-06-07 16:01:06.302009: I tensorflow/stream_executor/platform/default/dso_loader.cc:44] Successfully opened dynamic library libcusparse.so.10
2020-06-07 16:01:06.800506: I tensorflow/stream_executor/platform/default/dso_loader.cc:44] Successfully opened dynamic library libcudnn.so.7
2020-06-07 16:01:06.800720: I tensorflow/stream_executor/cuda/cuda_gpu_executor.cc:981] successful NUMA node read from SysFS had negative value (-1), but there must be at least one NUMA node, so returning NUMA node zero
2020-06-07 16:01:06.801460: I tensorflow/stream_executor/cuda/cuda_gpu_executor.cc:981] successful NUMA node read from SysFS had negative value (-1), but there must be at least one NUMA node, so returning NUMA node zero
2020-06-07 16:01:06.801942: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1703] Adding visible gpu devices: 0
2020-06-07 16:01:06.821596: I tensorflow/core/platform/profile_utils/cpu_utils.cc:102] CPU Frequency: 2300000000 Hz
2020-06-07 16:01:06.821888: I tensorflow/compiler/xla/service/service.cc:168] XLA service 0x310f100 initialized for platform Host (this does not guarantee that XLA will be used). Devices:
2020-06-07 16:01:06.821919: I tensorflow/compiler/xla/service/service.cc:176]   StreamExecutor device (0): Host, Default Version
2020-06-07 16:01:06.958871: I tensorflow/stream_executor/cuda/cuda_gpu_executor.cc:981] successful NUMA node read from SysFS had negative value (-1), but there must be at least one NUMA node, so returning NUMA node zero
2020-06-07 16:01:06.959547: I tensorflow/compiler/xla/service/service.cc:168] XLA service 0x310ef40 initialized for platform CUDA (this does not guarantee that XLA will be used). Devices:
2020-06-07 16:01:06.959585: I tensorflow/compiler/xla/service/service.cc:176]   StreamExecutor device (0): Tesla P100-PCIE-16GB, Compute Capability 6.0
2020-06-07 16:01:06.960661: I tensorflow/stream_executor/cuda/cuda_gpu_executor.cc:981] successful NUMA node read from SysFS had negative value (-1), but there must be at least one NUMA node, so returning NUMA node zero
2020-06-07 16:01:06.961233: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1561] Found device 0 with properties: 
pciBusID: 0000:00:04.0 name: Tesla P100-PCIE-16GB computeCapability: 6.0
coreClock: 1.3285GHz coreCount: 56 deviceMemorySize: 15.90GiB deviceMemoryBandwidth: 681.88GiB/s
2020-06-07 16:01:06.961280: I tensorflow/stream_executor/platform/default/dso_loader.cc:44] Successfully opened dynamic library libcudart.so.10.1
2020-06-07 16:01:06.961317: I tensorflow/stream_executor/platform/default/dso_loader.cc:44] Successfully opened dynamic library libcublas.so.10
2020-06-07 16:01:06.961332: I tensorflow/stream_executor/platform/default/dso_loader.cc:44] Successfully opened dynamic library libcufft.so.10
2020-06-07 16:01:06.961343: I tensorflow/stream_executor/platform/default/dso_loader.cc:44] Successfully opened dynamic library libcurand.so.10
2020-06-07 16:01:06.961356: I tensorflow/stream_executor/platform/default/dso_loader.cc:44] Successfully opened dynamic library libcusolver.so.10
2020-06-07 16:01:06.961366: I tensorflow/stream_executor/platform/default/dso_loader.cc:44] Successfully opened dynamic library libcusparse.so.10
2020-06-07 16:01:06.961380: I tensorflow/stream_executor/platform/default/dso_loader.cc:44] Successfully opened dynamic library libcudnn.so.7
2020-06-07 16:01:06.961476: I tensorflow/stream_executor/cuda/cuda_gpu_executor.cc:981] successful NUMA node read from SysFS had negative value (-1), but there must be at least one NUMA node, so returning NUMA node zero
2020-06-07 16:01:06.962009: I tensorflow/stream_executor/cuda/cuda_gpu_executor.cc:981] successful NUMA node read from SysFS had negative value (-1), but there must be at least one NUMA node, so returning NUMA node zero
2020-06-07 16:01:06.962525: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1703] Adding visible gpu devices: 0
2020-06-07 16:01:06.965663: I tensorflow/stream_executor/platform/default/dso_loader.cc:44] Successfully opened dynamic library libcudart.so.10.1
2020-06-07 16:01:13.402219: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1102] Device interconnect StreamExecutor with strength 1 edge matrix:
2020-06-07 16:01:13.402283: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1108]      0 
2020-06-07 16:01:13.402292: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1121] 0:   N 
2020-06-07 16:01:13.407188: I tensorflow/stream_executor/cuda/cuda_gpu_executor.cc:981] successful NUMA node read from SysFS had negative value (-1), but there must be at least one NUMA node, so returning NUMA node zero
2020-06-07 16:01:13.407806: I tensorflow/stream_executor/cuda/cuda_gpu_executor.cc:981] successful NUMA node read from SysFS had negative value (-1), but there must be at least one NUMA node, so returning NUMA node zero
2020-06-07 16:01:13.408301: W tensorflow/core/common_runtime/gpu/gpu_bfc_allocator.cc:39] Overriding allow_growth setting because the TF_FORCE_GPU_ALLOW_GROWTH environment variable is set. Original config value was 0.
2020-06-07 16:01:13.408344: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1247] Created TensorFlow device (/job:localhost/replica:0/task:0/device:GPU:0 with 14974 MB memory) -> physical GPU (device: 0, name: Tesla P100-PCIE-16GB, pci bus id: 0000:00:04.0, compute capability: 6.0)
total examples: 71235, batch_size: 64, epochs: 20, steps: 22280

Epoch 0
2020-06-07 16:01:17.898579: I tensorflow/stream_executor/platform/default/dso_loader.cc:44] Successfully opened dynamic library libcublas.so.10
2020-06-07 16:01:19.371652: I tensorflow/stream_executor/platform/default/dso_loader.cc:44] Successfully opened dynamic library libcudnn.so.7
step 100, loss:2.283, accuracy:0.150
step 200, loss:2.190, accuracy:0.310
step 300, loss:1.975, accuracy:0.391
step 400, loss:1.626, accuracy:0.489
step 500, loss:1.328, accuracy:0.563
step 600, loss:1.136, accuracy:0.642
step 700, loss:0.935, accuracy:0.700
step 800, loss:0.802, accuracy:0.728
step 900, loss:0.717, accuracy:0.748
step 1000, loss:0.640, accuracy:0.768
step 1100, loss:0.596, accuracy:0.781

Epoch 1
step 1200, loss:0.552, accuracy:0.798
step 1300, loss:0.532, accuracy:0.793
step 1400, loss:0.490, accuracy:0.813
step 1500, loss:0.472, accuracy:0.821
step 1600, loss:0.476, accuracy:0.814
step 1700, loss:0.462, accuracy:0.814
step 1800, loss:0.418, accuracy:0.836
step 1900, loss:0.428, accuracy:0.827
step 2000, loss:0.406, accuracy:0.842
step 2100, loss:0.426, accuracy:0.824
step 2200, loss:0.409, accuracy:0.833

Epoch 2
step 2300, loss:0.387, accuracy:0.844
step 2400, loss:0.375, accuracy:0.850
step 2500, loss:0.367, accuracy:0.848
step 2600, loss:0.366, accuracy:0.848
step 2700, loss:0.359, accuracy:0.848
step 2800, loss:0.351, accuracy:0.852
step 2900, loss:0.360, accuracy:0.852
step 3000, loss:0.364, accuracy:0.849
step 3100, loss:0.347, accuracy:0.853
step 3200, loss:0.350, accuracy:0.849
step 3300, loss:0.349, accuracy:0.848

Epoch 3
step 3400, loss:0.338, accuracy:0.855
step 3500, loss:0.317, accuracy:0.870
step 3600, loss:0.320, accuracy:0.862
step 3700, loss:0.311, accuracy:0.865
step 3800, loss:0.312, accuracy:0.864
step 3900, loss:0.314, accuracy:0.862
step 4000, loss:0.308, accuracy:0.868
step 4100, loss:0.315, accuracy:0.862
step 4200, loss:0.315, accuracy:0.864
step 4300, loss:0.308, accuracy:0.867
step 4400, loss:0.309, accuracy:0.868

Epoch 4
step 4500, loss:0.302, accuracy:0.867
step 4600, loss:0.275, accuracy:0.878
step 4700, loss:0.290, accuracy:0.867
step 4800, loss:0.288, accuracy:0.870
step 4900, loss:0.298, accuracy:0.867
step 5000, loss:0.289, accuracy:0.871
step 5100, loss:0.281, accuracy:0.877
step 5200, loss:0.285, accuracy:0.872
step 5300, loss:0.272, accuracy:0.876
step 5400, loss:0.281, accuracy:0.868
step 5500, loss:0.285, accuracy:0.875

Epoch 5
step 5600, loss:0.277, accuracy:0.876
step 5700, loss:0.259, accuracy:0.879
step 5800, loss:0.258, accuracy:0.882
step 5900, loss:0.258, accuracy:0.882
step 6000, loss:0.261, accuracy:0.883
step 6100, loss:0.257, accuracy:0.882
step 6200, loss:0.260, accuracy:0.881
step 6300, loss:0.270, accuracy:0.873
step 6400, loss:0.268, accuracy:0.879
step 6500, loss:0.272, accuracy:0.870
step 6600, loss:0.259, accuracy:0.874

Epoch 6
step 6700, loss:0.265, accuracy:0.878
step 6800, loss:0.243, accuracy:0.889
step 6900, loss:0.242, accuracy:0.887
step 7000, loss:0.248, accuracy:0.881
step 7100, loss:0.239, accuracy:0.889
step 7200, loss:0.243, accuracy:0.885
step 7300, loss:0.246, accuracy:0.885
step 7400, loss:0.250, accuracy:0.880
step 7500, loss:0.238, accuracy:0.885
step 7600, loss:0.241, accuracy:0.883
step 7700, loss:0.242, accuracy:0.885

Epoch 7
step 7800, loss:0.256, accuracy:0.879
step 7900, loss:0.228, accuracy:0.897
step 8000, loss:0.224, accuracy:0.895
step 8100, loss:0.216, accuracy:0.899
step 8200, loss:0.220, accuracy:0.896
step 8300, loss:0.232, accuracy:0.891
step 8400, loss:0.235, accuracy:0.886
step 8500, loss:0.220, accuracy:0.897
step 8600, loss:0.237, accuracy:0.876
step 8700, loss:0.231, accuracy:0.882
step 8800, loss:0.234, accuracy:0.888
step 8900, loss:0.230, accuracy:0.889

Epoch 8
step 9000, loss:0.211, accuracy:0.901
step 9100, loss:0.212, accuracy:0.900
step 9200, loss:0.212, accuracy:0.897
step 9300, loss:0.207, accuracy:0.902
step 9400, loss:0.209, accuracy:0.895
step 9500, loss:0.213, accuracy:0.897
step 9600, loss:0.217, accuracy:0.892
step 9700, loss:0.226, accuracy:0.889
step 9800, loss:0.217, accuracy:0.890
step 9900, loss:0.229, accuracy:0.889
step 10000, loss:0.220, accuracy:0.889

Epoch 9
step 10100, loss:0.197, accuracy:0.900
step 10200, loss:0.198, accuracy:0.905
step 10300, loss:0.202, accuracy:0.896
step 10400, loss:0.200, accuracy:0.902
step 10500, loss:0.206, accuracy:0.899
step 10600, loss:0.211, accuracy:0.895
step 10700, loss:0.208, accuracy:0.894
step 10800, loss:0.205, accuracy:0.893
step 10900, loss:0.198, accuracy:0.901
step 11000, loss:0.205, accuracy:0.900
step 11100, loss:0.209, accuracy:0.891

Epoch 10
step 11200, loss:0.194, accuracy:0.902
step 11300, loss:0.181, accuracy:0.912
step 11400, loss:0.179, accuracy:0.911
step 11500, loss:0.190, accuracy:0.902
step 11600, loss:0.201, accuracy:0.897
step 11700, loss:0.193, accuracy:0.904
step 11800, loss:0.193, accuracy:0.900
step 11900, loss:0.202, accuracy:0.895
step 12000, loss:0.201, accuracy:0.894
step 12100, loss:0.196, accuracy:0.898
step 12200, loss:0.204, accuracy:0.895

Epoch 11
step 12300, loss:0.192, accuracy:0.905
step 12400, loss:0.183, accuracy:0.908
step 12500, loss:0.177, accuracy:0.913
step 12600, loss:0.175, accuracy:0.911
step 12700, loss:0.190, accuracy:0.902
step 12800, loss:0.194, accuracy:0.899
step 12900, loss:0.183, accuracy:0.908
step 13000, loss:0.192, accuracy:0.900
step 13100, loss:0.194, accuracy:0.897
step 13200, loss:0.187, accuracy:0.908
step 13300, loss:0.199, accuracy:0.898

Epoch 12
step 13400, loss:0.186, accuracy:0.900
step 13500, loss:0.176, accuracy:0.909
step 13600, loss:0.179, accuracy:0.906
step 13700, loss:0.175, accuracy:0.911
step 13800, loss:0.169, accuracy:0.914
step 13900, loss:0.184, accuracy:0.901
step 14000, loss:0.183, accuracy:0.900
step 14100, loss:0.181, accuracy:0.910
step 14200, loss:0.177, accuracy:0.906
step 14300, loss:0.186, accuracy:0.901
step 14400, loss:0.189, accuracy:0.898

Epoch 13
step 14500, loss:0.183, accuracy:0.904
step 14600, loss:0.167, accuracy:0.916
step 14700, loss:0.177, accuracy:0.907
step 14800, loss:0.177, accuracy:0.904
step 14900, loss:0.173, accuracy:0.911
step 15000, loss:0.177, accuracy:0.902
step 15100, loss:0.171, accuracy:0.914
step 15200, loss:0.178, accuracy:0.900
step 15300, loss:0.176, accuracy:0.907
step 15400, loss:0.177, accuracy:0.905
step 15500, loss:0.176, accuracy:0.906

Epoch 14
step 15600, loss:0.175, accuracy:0.906
step 15700, loss:0.159, accuracy:0.913
step 15800, loss:0.162, accuracy:0.915
step 15900, loss:0.168, accuracy:0.910
step 16000, loss:0.170, accuracy:0.911
step 16100, loss:0.178, accuracy:0.904
step 16200, loss:0.177, accuracy:0.902
step 16300, loss:0.172, accuracy:0.904
step 16400, loss:0.175, accuracy:0.902
step 16500, loss:0.171, accuracy:0.905
step 16600, loss:0.174, accuracy:0.913
step 16700, loss:0.167, accuracy:0.908

Epoch 15
step 16800, loss:0.155, accuracy:0.916
step 16900, loss:0.163, accuracy:0.913
step 17000, loss:0.165, accuracy:0.909
step 17100, loss:0.164, accuracy:0.909
step 17200, loss:0.166, accuracy:0.913
step 17300, loss:0.164, accuracy:0.914
step 17400, loss:0.171, accuracy:0.905
step 17500, loss:0.171, accuracy:0.910
step 17600, loss:0.165, accuracy:0.913
step 17700, loss:0.167, accuracy:0.909
step 17800, loss:0.173, accuracy:0.900

Epoch 16
step 17900, loss:0.167, accuracy:0.910
step 18000, loss:0.159, accuracy:0.915
step 18100, loss:0.158, accuracy:0.915
step 18200, loss:0.167, accuracy:0.906
step 18300, loss:0.161, accuracy:0.913
step 18400, loss:0.166, accuracy:0.908
step 18500, loss:0.161, accuracy:0.911
step 18600, loss:0.162, accuracy:0.914
step 18700, loss:0.160, accuracy:0.911
step 18800, loss:0.162, accuracy:0.912
step 18900, loss:0.160, accuracy:0.910

Epoch 17
step 19000, loss:0.158, accuracy:0.914
step 19100, loss:0.149, accuracy:0.920
step 19200, loss:0.158, accuracy:0.915
step 19300, loss:0.161, accuracy:0.910
step 19400, loss:0.157, accuracy:0.918
step 19500, loss:0.166, accuracy:0.905
step 19600, loss:0.151, accuracy:0.919
step 19700, loss:0.162, accuracy:0.907
step 19800, loss:0.167, accuracy:0.909
step 19900, loss:0.159, accuracy:0.908
step 20000, loss:0.165, accuracy:0.904

Epoch 18
step 20100, loss:0.152, accuracy:0.918
step 20200, loss:0.155, accuracy:0.918
step 20300, loss:0.155, accuracy:0.917
step 20400, loss:0.159, accuracy:0.910
step 20500, loss:0.157, accuracy:0.915
step 20600, loss:0.155, accuracy:0.914
step 20700, loss:0.156, accuracy:0.914
step 20800, loss:0.154, accuracy:0.912
step 20900, loss:0.154, accuracy:0.915
step 21000, loss:0.157, accuracy:0.912
step 21100, loss:0.160, accuracy:0.905

Epoch 19
step 21200, loss:0.161, accuracy:0.916
step 21300, loss:0.149, accuracy:0.918
step 21400, loss:0.162, accuracy:0.915
step 21500, loss:0.150, accuracy:0.920
step 21600, loss:0.156, accuracy:0.910
step 21700, loss:0.157, accuracy:0.913
step 21800, loss:0.152, accuracy:0.913
step 21900, loss:0.149, accuracy:0.918
step 22000, loss:0.151, accuracy:0.915
step 22100, loss:0.161, accuracy:0.909
step 22200, loss:0.151, accuracy:0.912
PCNN model saved: ./output/pcnn.ckpt-22280
Restoring model from ./output/pcnn.ckpt-22280...

(['the', 'cheering', 'was', 'loudest', 'when', 'mr.', 'bush', ',', 'flanked', 'by', 'the', 'two', 'top', 'americans', 'in', 'baghdad', ',', 'gen.', 'george', 'w.', 'casey', 'jr.', ',', 'the', 'overall', 'military', 'commander', ',', 'and', 'ambassador', 'zalmay', 'khalilzad', ',', 'spoke', 'of', 'saddam', 'hussein', 'as', 'a', "''", 'selfish', ',', 'brutal', 'leader', "''", 'who', 'had', 'humiliated', 'iraqis', 'and', 'denied', 'them', 'freedom', ',', 'and', 'when', 'he', 'invoked', 'the', 'bombing', 'strike', 'last', 'week', 'that', 'killed', 'america', "'s", 'most-wanted', 'man', 'in', 'iraq', ',', 'abu', 'musab', 'al-zarqawi', '.', "''", '</s>'], 'iraq', 'baghdad')
  prediction/probability:/location/country/capital/0.622, target:/location/country/capital

(['n.h.l.', 'buffalo', 'sabres', '--', 'assigned', 'f', 'daniel_paille', ',', 'c', 'chris_taylor', ',', 'd', 'david', 'cullen', ',', 'rw', 'sean', 'mcmorrow', ',', 'd', 'john', 'adams', ',', 'd', 'nathan', 'paetsch', ',', 'f', 'branislav', 'fabry', ',', 'f', 'dylan', 'hunter', ',', 'f', 'clarke', 'macarthur', ',', 'f', 'mark', 'mancari', ',', 'f', 'jiri', 'novotny', ',', 'f', 'michael', 'ryan', ',', 'f', 'chris', 'thorburn', 'and', 'g', 'scott', 'stirling', 'to', 'rochester', 'of', 'the', 'ahl', '.', '</s>'], 'chris_taylor', 'daniel_paille')
  prediction/probability:NA/1.000, target:NA

(['marty', 'golden', ',', 'a', 'new', 'york', 'state', 'senator', ',', 'asked', 'dr.', 'narmesh', 'shah', 'on', 'a', 'recent', 'summer', 'day', ',', 'walking', 'into', 'a', 'pizza', 'parlor', 'next', 'to', 'golden', "'s", 'brooklyn', 'office', 'in', 'the', '22nd', 'district', 'in', 'bay_ridge', '.', '</s>'], 'bay_ridge', 'brooklyn')
  prediction/probability:/location/neighborhood/neighborhood_of/0.995, target:/location/neighborhood/neighborhood_of

(['senators', 'evan_bayh', 'of', 'indiana', ',', 'hillary', 'rodham', 'clinton', 'of', 'new', 'york', 'and', 'barack_obama', 'of', 'illinois', 'are', 'among', 'those', 'quietly', 'making', 'preparations', 'for', 'possible', 'campaigns', ',', 'as', 'is', 'former', 'senator', 'john', 'edwards', 'of', 'north', 'carolina', '.', '</s>'], 'barack_obama', 'evan_bayh')
  prediction/probability:NA/0.995, target:NA

(['having', 'vowed', 'to', 'a', 'sardinian', 'priest', 'that', 'he', 'would', 'not', 'have', 'sex', 'until', 'after', 'the', 'april', '9', 'election', ',', 'prime', 'minister', 'silvio_berlusconi', 'spent', 'the', 'first', 'post-pledge', 'day', 'at', 'a', 'birthday', 'party', 'in', 'milan', 'for', 'his', 'mother', ',', 'rosa', 'bossi', ',', 'who', 'turned', '95', '.', '</s>'], 'silvio_berlusconi', 'milan')
  prediction/probability:/people/person/place_lived/0.998, target:/people/person/place_lived

(['a1', 'marla', 'ruzicka', ',', 'a', 'fiercely', 'energetic', '28-year-old', 'from', 'california', 'who', 'ran', 'a', 'one-woman', 'aid', 'mission', 'in', 'iraq', ',', 'was', 'killed', 'by', 'a', 'suicide', 'bomber', ',', 'u.s.', 'embassy', 'officials', 'in', 'baghdad', 'said', '.', '</s>'], 'iraq', 'baghdad')
  prediction/probability:/location/country/capital/0.649, target:/location/country/capital

(['over', 'the', 'last', 'few', 'years', ',', 'the', 'owner', 'of', 'a', 'century-old', 'brownstone', 'in', 'park_slope', ',', 'brooklyn', ',', 'has', 'emerged', 'relatively', 'unscathed', ',', 'because', 'state', 'tax', 'law', 'is', 'particularly', 'favorable', 'to', 'older', 'houses', 'in', 'neighborhoods', 'where', 'prices', 'have', 'risen', 'quickly', '.', '</s>'], 'park_slope', 'brooklyn')
  prediction/probability:/location/neighborhood/neighborhood_of/1.000, target:/location/neighborhood/neighborhood_of

(['stephen_king', "'s", 'maine', 'chronicles', 'have', 'underscored', 'the', 'point', '.', '</s>'], 'stephen_king', 'maine')
  prediction/probability:/people/person/place_lived/1.000, target:/people/person/place_lived

(['vice', 'president', 'dick_cheney', 'formerly', 'headed', 'halliburton', ',', 'a', 'conglomerate', 'based', 'in', 'texas', '.', '</s>'], 'dick_cheney', 'halliburton')
  prediction/probability:/business/person/company/0.984, target:/business/person/company

(['bush', 'administration', 'officials', 'say', 'they', 'recognize', 'that', 'in', 'the', 'new', 'democratic-controlled', 'congress', ',', 'anger', 'at', 'china', 'will', 'quickly', 'reach', 'a', 'fever', 'pitch', 'if', 'the', 'beijing', 'government', 'rebuffs', 'american', 'requests', 'for', 'steps', 'to', 'ease', 'the', 'trade', 'imbalances', '.', '</s>'], 'beijing', 'china')
  prediction/probability:/location/administrative_division/country/1.000, target:/location/administrative_division/country

(['indonesia', ',', 'where', 'the', 'tsunami', 'on', 'dec.', '26', 'originated', ',', 'had', 'a', 'smaller', 'worry', 'four', 'years', 'ago', 'when', 'the', 'world', 'bridge', 'federation', 'moved', 'the', 'world', 'championship', 'from', 'bali', 'to', 'paris', 'at', 'the', 'last', 'minute', '.', '</s>'], 'indonesia', 'bali')
  prediction/probability:/location/country/administrative_divisions/0.627, target:/location/country/administrative_divisions

(['in', 'the', 'day', "'s", 'other', 'surprising', 'news', ',', 'walker', ',', 'a', 'forward', ',', 'returned', 'to', 'the', 'boston_celtics', 'two', 'years', 'after', 'danny_ainge', ',', 'the', 'team', "'s", 'executive', 'director', 'of', 'basketball', 'operations', ',', 'traded', 'him', '.', '</s>'], 'danny_ainge', 'boston_celtics')
  prediction/probability:/business/person/company/0.991, target:/business/person/company

(['the', 'topic', 'is', 'likely', 'to', 'be', 'near', 'the', 'top', 'of', 'the', 'agenda', 'when', 'president', 'hu_jintao', 'of', 'china', 'meets', 'president', 'bush', 'in', 'washington', 'on', 'thursday', '.', '</s>'], 'hu_jintao', 'china')
  prediction/probability:/people/person/nationality/0.998, target:/people/person/nationality

(['during', 'the', 'war', 'between', 'iraq', 'and', 'iran', 'in', 'the', '1980', "'s", ',', 'saudi_arabia', 'had', 'given', 'iraq', '$', '25', 'billion', 'in', 'aid', '.', '</s>'], 'saudi_arabia', 'iraq')
  prediction/probability:NA/0.926, target:NA

(['there', 'were', 'more', 'than', 'a', 'few', 'strange', 'and', 'riveting', 'moments', ',', 'including', 'hideki_matsui', 'of', 'the', 'yankees', 'sustaining', 'a', 'broken', 'wrist', ',', 'which', 'ended', 'his', 'consecutive-games', 'streak', 'in', 'the', 'major', 'leagues', 'and', 'japan', 'at', '1,768', '.', '</s>'], 'hideki_matsui', 'japan')
  prediction/probability:/people/person/nationality/1.000, target:/people/person/nationality

(['to', 'the', 'editor', ':', 're', "''", 'pointless', 'provocation', 'in', 'tokyo', "''", '-lrb-', 'editorial', ',', 'oct.', '18', '-rrb-', ',', 'about', 'prime', 'minister', 'junichiro', 'koizumi', "'s", 'visit', 'to', 'the', 'yasukuni', 'shrine', ':', 'mr.', 'koizumi', "'s", 'visit', 'was', 'not', 'to', 'worship', 'the', 'class', 'a', 'war', 'criminals', 'who', 'were', 'given', 'the', 'verdict', 'of', 'guilty', 'by', 'the', 'international', 'tribunal', 'for', 'the', 'far', 'east', ',', 'to', 'glorify', 'japan', "'s", 'past', 'militarism', ',', 'or', 'to', 'accommodate', 'right-wing', 'nationalists', '.', '</s>'], 'japan', 'tokyo')
  prediction/probability:/location/country/administrative_divisions/0.544, target:/location/country/administrative_divisions

(['lindsay', ',', 'who', ',', 'like', 'mr.', 'miller', 'represented', 'manhattan', "'s", 'upper_east_side', '-lrb-', 'it', 'was', 'called', 'the', 'silk', 'stocking', 'district', 'then', '-rrb-', ',', 'described', 'white', 'anglo-saxon', 'protestants', 'as', 'an', 'endangered', 'species', 'and', 'added', ':', "''", 'you', 'realize', 'you', 'ca', "n't", 'call', 'any', 'other', 'ethnic', 'group', 'by', 'its', 'pejorative', 'name', '.', '</s>'], 'upper_east_side', 'manhattan')
  prediction/probability:/location/neighborhood/neighborhood_of/1.000, target:/location/neighborhood/neighborhood_of

(['cablevision', ',', 'based', 'in', 'bethpage', ',', 'n.y.', ',', 'has', 'lucrative', 'systems', 'in', 'new_york', "'s", 'suburbs', ',', 'while', 'adelphia', 'has', 'systems', 'in', 'los', 'angeles', ',', 'upstate', 'new_york', 'and', 'elsewhere', 'in', 'the', 'country', '.', '</s>'], 'new_york', 'bethpage')
  prediction/probability:/location/location/contains/0.988, target:/location/location/contains

(['a', 'european', 'diplomat', 'in', 'beijing', 'said', 'last', 'week', 'that', 'the', 'anti-secession', 'bill', ',', 'especially', 'if', 'it', 'prompts', 'a', 'tit-for-tat', 'response', 'from', 'taiwan', ',', 'could', 'raise', 'the', 'risk', 'of', 'conflict', 'and', 'cause', 'the', 'european', 'union', 'to', 'delay', 'the', 'lifting', 'of', 'its', 'arms', 'embargo', 'on', 'china', ',', 'one', 'of', 'beijing', "'s", 'top', 'priorities', '.', '</s>'], 'beijing', 'china')
  prediction/probability:/location/administrative_division/country/1.000, target:/location/administrative_division/country

(['i', 'used', 'to', 'joke', 'around', ',', 'everyone', 'kind', 'of', 'lives', 'in', 'the', 'mooresville', 'area', 'in', 'north_carolina', 'and', 'i', 'would', 'call', 'it', '`', 'as', 'mooresville', 'turns', '.', "'", '</s>'], 'north_carolina', 'mooresville')
  prediction/probability:/location/location/contains/0.999, target:/location/location/contains
```

