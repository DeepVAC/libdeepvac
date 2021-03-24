# libdeepvac
Use PyTorch model in C++ project.  

这个项目定义了如何在C++项目中使用PyTorch训练的模型。

# 简介

在MLab（云上炼丹师）实验室，我们使用[DeepVAC](https://github.com/DeepVAC/deepvac) 来训练获得新模型，使用本项目来部署模型。

libdeepvac作为一个Linux库，在以下四个方面发挥了价值：
- 向下封装了推理引擎，目前封装了LibTorch，即将封装TensorRT、NCNN；
- 向上提供Deepvac类，方便用户继承并实现其自定义的模型；
- 在modules目录下，MLab提供了经典网络的C++实现；
- 在utils目录下，MLab提供了网络中常见helper函数的C++实现。

# libdeepvac实现的模块
### SOTA网络的C++实现
|类名 | 网络 | 作用 |
|-----|------|------|
|SyszuxFaceRetina| RetinaNet | 人脸检测|
|SyszuxOcrPse | PSENet | 文字检测 |
|SyszuxSegEsp | ESPNetV2 | 语义分割 |
|SyszuxClsMobile | MobileNetV3 | 分类 |
|SyszuxDetectYolo | YOLOV5 | 目标检测 |
|SyszuxClsResnet | ResNet50 | 分类 |

### helper函数实现
|类名/函数名 | 作用 |
|-----|------|
|AlignFace|人脸对齐|
|nms | 检测框的非极大值抑制|
|PriorBox| 生成目标检测的候选框 |


未来我们会持续在modules、utils目录下提供SOTA网络的C++实现。如果用户（你）需要什么网络的C++实现，可在issues里提交申请。

# 编译平台的支持
libdeepvac支持在以下平台上进行编译：
- x86_64 GNU/Linux（或者叫 AMD64 GNU/Linux）
- aarch64 GNU/Linux（或者叫 ARM64 GNU/Linux）
- macOS  

未来不太可能扩展到其它平台。


# 编译目标的支持
libdeepvac支持以下目标平台的编译：
- x86_64 GNU/Linux（或者叫 AMD64 GNU/Linux）
- x86_64 GNU/Linux with CUDA（或者叫 AMD64 GNU/Linux with CUDA）
- aarch64 GNU/Linux（或者叫 ARM64 GNU/Linux）
- Android
- iOS
- Nvidia Jetson Xavier NX（Volta,384 CUDA cores, 48 Tensor cores, 6-core, 8GB）
- Nvidia Jetson AGX Xavier (Volta, 512 CUDA cores, 6-core, 32GB )
- Nvidia Jetson TX2 (Pascal, 256 CUDA cores, 2-core/4-core, 8GB)
- Nvidia Jetson TX2 NX (Pascal, 256 CUDA cores, 2-core/4-core, 4GB)

# 项目依赖
libdeepvac的编译依赖C++14编译器、CMake、opencv、LibTorch。  
最简便、高效的方式就是使用我们提供的DeepVAC开发时[Docker镜像](https://github.com/DeepVAC/deepvac#2-%E7%8E%AF%E5%A2%83%E5%87%86%E5%A4%87)。

- 该镜像内置的LibTorch版本为：[LibTorch4DeepVAC (x86-64 Linux) 1.8.0](https://github.com/CivilNet/libtorch/releases/download/1.8.0/libtorch.tar.gz)；
- 你也可以使用[PyTorch官方](https://pytorch.org/)LibTorch版本替换内置的LibTorch版本；
- 未来几个月，该镜像还将包含LibTorch4DeepVAC (Android) 1.8.0；


# 编译开关
如果要开始编译libdeepvac，需要先熟悉如下几个编译开关的作用：
- BUILD_STATIC，默认打开。是否静态编译libdeepvac库；
- USE_MKL，默认关闭。是否使用MKL来作为BLAS/LAPACK后端；
- USE_CUDA，默认关闭。是否使用CUDA；
- USE_NUMA，默认关闭。是否使用libnuma库（仅在开启CUDA后有意义）；
- USE_TENSORRT，默认关闭。是否使用TensorRT；
- USE_LOADER，默认关闭。是否使用图片装载器，需要C++17编译器；
- GARRULOUS_GEMFIELD，默认关闭。是否打开调试log；
- BUILD_ALL_EXAMPLES，默认关闭。是否编译所有的examples；
- SYSTEM_LAPACK_LIBRARIES，默认为空。指定额外的BLAS/LAPACK库。

CMake命令如下：

```bash
# create build directory
mkdir build
cd build

#比如在X86_64 GPU服务器上，使用CUDA，且用MKL作为BLAS/LAPACK库
cmake -DUSE_MKL=ON -DUSE_CUDA=ON -DCMAKE_BUILD_TYPE=Release -DCMAKE_PREFIX_PATH="/gemfield/libtorch;/gemfield/opencv4deepvac/" -DCMAKE_INSTALL_PREFIX=../install .. 

#比如在Nvidia Jetson Xavier NX上，使用TensorRT，且用系统的blas和lapack库
cmake -DUSE_CUDA=ON -DUSE_NUMA=ON -DUSE_TENSORRT=ON -DSYSTEM_LAPACK_LIBRARIES="-lblas -llapack" -DCMAKE_BUILD_TYPE=Release -DCMAKE_PREFIX_PATH="/home/gemfield/github/opencv4deepvac/;/home/gemfield/github/libtorch" -DCMAKE_INSTALL_PREFIX=../install ..


cmake --build . --config Release
make install
```

# 使用libdeepvac预编译库
如何在自己的项目中使用libdeepvac预编译库呢？
## 1. 添加find_package(Deepvac REQUIRED)
在自己项目的CMakeLists.txt中，添加
```
find_package(Deepvac REQUIRED)
```

## 2. 使用libdeepvac提供的头文件cmake变量
在自己项目的CMakeLists.txt中，你可以使用如下cmake变量：
- DEEPVAC_INCLUDE_DIRS：libdeepvac库的头文件目录；
- DEEPVAC_LIBTORCH_INCLUDE_DIRS：libtorch库的头文件目录；
- DEEPVAC_TENSORRT_INCLUDE_DIRS：TensorRT库的头文件目录；
- DEEPVAC_CV_INCLUDE_DIRS：OpenCV库的头文件目录；

## 3. 使用libdeepvac提供的库文件cmake变量
- DEEPVAC_LIBRARIES：libdeepvac库；
- DEEPVAC_LIBTORCH_CPU_LIBRARIES：libtorch cpu版库；
- DEEPVAC_LIBTORCH_CUDA_LIBRARIES：libtorch cuda版库；
- DEEPVAC_LIBTORCH_DEFAULT_LIBRARIES：libtorch默认版库（编译时用的cpu还是cuda）；
- DEEPVAC_LIBCUDA_LIBRARIES：Nvidia cuda runtime库；
- DEEPVAC_TENSORRT_LIBRARIES：Nvidia TensorRT runtime库；
- DEEPVAC_CV_LIBRARIES：OpenCV库；

使用举例：
```
#头文件
target_include_directories(${your_target} "${DEEPVAC_LIBTORCH_INCLUDE_DIRS};${DEEPVAC_TENSORRT_INCLUDE_DIRS};${CMAKE_CURRENT_SOURCE_DIR}/include>")

#库文件
target_link_libraries( ${your_target} ${DEEPVAC_LIBRARIES} ${DEEPVAC_LIBTORCH_CUDA_LIBRARIES} ${DEEPVAC_LIBCUDA_LIBRARIES} ${DEEPVAC_CV_LIBRARIES})
```
# 演示
[SYSZUX-FACE](https://github.com/CivilNet/SYSZUX-FACE)基于本项目实现了人脸检测功能。

