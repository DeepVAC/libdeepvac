# libdeepvac
Use PyTorch model in C++ project.  

这个项目定义了如何在C++项目中使用PyTorch训练的模型。

# 简介

在MLab（云上炼丹师）实验室，我们使用[DeepVAC](https://github.com/DeepVAC/deepvac) 来训练获得新模型，使用本项目来部署模型。

libdeepvac作为一个Linux库，在以下四个方面发挥了价值：
- 向下封装了推理引擎，目前封装了LibTorch，即将封装TensorRT、NCNN、TNN；
- 向上提供Deepvac类，方便用户继承并实现其自定义的模型；
- 在modules目录下，MLab提供了经典网络的C++实现；
- 在utils目录下，MLab提供了网络中常见helper函数的C++实现。

# libdeepvac实现的模块
### SOTA网络的C++实现
|类名 | 网络 | 作用 |
|-----|------|------|
|SyszuxFaceRetina| RetinaNet | 人脸检测|
|SyszuxOcrPse | PSENet | 文字检测 |
|SyszuxOcrDB  | DB Net | 文字检测 |
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
最简便、高效的方式就是使用我们提供的[MLab HomePod](https://github.com/DeepVAC/MLab#mlab-homepod)。使用MLab HomePod也是我们推荐的方式。

# 如何编译libdeepvac
libdeepvac基于CMake进行构建。

## 编译开关
如果要开始编译libdeepvac，需要先熟悉如下几个CMake选项的作用：
|CMake选项|默认值|常用值| 作用|备注|
|--------|-----|-----|-----|---|
|BUILD_STATIC|ON|ON/OFF| ON：编译静态libdeepvac<br>OFF: 编译动态libdeepvac|OFF时，链接OpenCV静态库会带来hidden symbol问题，此时需链接OpenCV动态库|
|USE_STATIC_LIBTORCH|OFF|ON/OFF|ON: 使用libtorch静态库<br>OFF: 使用libtorch动态库|MLab HomePod中内置有libtorch动态库|
|USE_MKL|OFF|ON/OFF| 是否使用Intel MKL作为LAPACK/BLAS实现|OFF的时候，需要使用SYSTEM_LAPACK_LIBRARIES指定另外的LAPACK/BLAS实现，比如openblas、Eigen等|
|SYSTEM_LAPACK_LIBRARIES|""|"-lblas -llapack"| USE_MKL关闭后需要指定的LAPACK/BLAS库|在系统路径下安装有相应的开发环境|
|USE_CUDA|OFF| ON/OFF| 是否使用CUDA|需要CUDA硬件，且系统中已经安装有CUDA ToolKit的开发时|
|USE_TENSORRT|OFF|ON/OFF|是否使用TensorRT|需要CUDA硬件，且系统中已经安装有TensorRT的开发时|
|USE_NUMA|OFF|ON/OFF| 是否链接-lnuma库|NA|
|USE_LOADER|OFF|ON/OFF| 是否使用图片装载器|需要C++17编译器|
|GARRULOUS_GEMFIELD|OFF|ON/OFF| 是否打开调试log|NA|
|BUILD_ALL_EXAMPLES|OFF|ON/OFF|是否编译所有的examples |NA|


## 下载依赖
**如果你使用的是MLab HomePod 2.0 pro（或者以上版本），则忽略此小节**。  
如果你使用的是自定义环境，那么你至少需要下载opencv库、libtorch库：
- opencv动态库：自行apt下载；
- opencv静态库：https://github.com/DeepVAC/libdeepvac/releases/download/1.9.0/opencv4deepvac.tar.gz 
- libtorch动态库：自行从PyTorch官网下载；
- libtorch静态库：https://github.com/DeepVAC/libdeepvac/releases/download/1.9.0/libtorch.tar.gz 

你亦可以在MLab HomePod 2.0 pro上自行从源码编译上述的依赖库。


## CMake命令
以下命令所使用路径均基于MLab HomePod 2.0 pro（你可以根据自身环境自行更改）。
#### 预备工作
```bash
# update to latest libdeepvac
gemfield@homepod2:/opt/gemfield/libdeepvac$ git pull --rebase
# create build directory
gemfield@homepod2:/opt/gemfield/libdeepvac$ mkdir build
gemfield@homepod2:/opt/gemfield/libdeepvac$ cd build
```

#### CMake
libdeepvac内置了诸多cmake开关以支持不同的软硬件开发栈：
- 在X86_64 GPU服务器上，使用CUDA，使用libtorch静态库，且用MKL作为BLAS/LAPACK库 (MLab HomePod 2.0 pro支持)：
```bash
cmake -DUSE_MKL=ON -DUSE_CUDA=ON -DUSE_STATIC_LIBTORCH=ON -DCMAKE_BUILD_TYPE=Release -DCMAKE_PREFIX_PATH="/opt/gemfield/libtorch;/opt/gemfield/opencv4deepvac/" -DCMAKE_INSTALL_PREFIX=../install ..
```
- 在X86_64 GPU服务器上，使用CUDA，使用libtorch动态库，且用MKL作为BLAS/LAPACK库 (MLab HomePod 2.0 pro支持)：
```bash
cmake -DUSE_MKL=ON -DUSE_CUDA=ON -DCMAKE_BUILD_TYPE=Release -DCMAKE_PREFIX_PATH="/opt/gemfield/opencv4deepvac;/opt/conda/lib/python3.8/site-packages/torch/" -DCMAKE_INSTALL_PREFIX=../install ..
```
- 在X86_64 GPU服务器上，使用TensorRT和libtorch静态库，且用MKL作为BLAS/LAPACK库：
```bash
cmake -DUSE_MKL=ON -DUSE_CUDA=ON -DUSE_MAGMA=ON -DUSE_STATIC_LIBTORCH=ON -DUSE_TENSORRT=ON -DCMAKE_BUILD_TYPE=Release -DCMAKE_PREFIX_PATH="/opt/gemfield/opencv4deepvac/;/opt/gemfield/libtorch" -DCMAKE_INSTALL_PREFIX=../install ..
```
- 在Nvidia Jetson Xavier NX上，使用TensorRT，且用系统的blas和lapack库：
```bash
cmake -DUSE_CUDA=ON -DUSE_NUMA=ON -DUSE_TENSORRT=ON -DSYSTEM_LAPACK_LIBRARIES="-lblas -llapack" -DCMAKE_BUILD_TYPE=Release -DCMAKE_PREFIX_PATH="/opt/gemfield/opencv4deepvac/;/opt/gemfield/libtorch" -DCMAKE_INSTALL_PREFIX=../install ..
```

#### 编译
```bash
cmake --build . --config Release
make install
```

# 如何使用libdeepvac库
如何在自己的项目中使用libdeepvac预编译库呢？
## 1. 添加find_package(Deepvac REQUIRED)
在自己项目的CMakeLists.txt中，添加
```cmake
find_package(Deepvac REQUIRED)
```
当然，基于libdeepvac的项目也必然基于opencv和libtorch，因此，下面两个find_package也是必须的：
```cmake
find_package(Torch REQUIRED)
find_package(OpenCV REQUIRED)
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

# Benchmark
libdeepvac会提供不同目标平台及不同推理引擎的benchmark，当前仅支持libtorch推理引擎。

## 1. X86-64 Linux + LibTorch的benchmark步骤
- 部署[MLab HomePod](https://github.com/DeepVAC/MLab#1-%E9%83%A8%E7%BD%B2);
- 克隆本项目：
```bash
# 如果是MLab HomePod 2.0 标准版
git clone https://github.com/DeepVAC/libdeepvac && cd libdeepvac

# 如果是MLab HomePod 2.0 pro版
cd /opt/gemfield/libdeepvac && git pull --rebase
```
- 编译
```bash
#新建编译目录
mkdir build
cd build
#cmake(如果基于LibTorch动态库)
cmake -DGARRULOUS_GEMFIELD=ON -DUSE_MKL=ON -DUSE_CUDA=ON -DCMAKE_BUILD_TYPE=Release -DCMAKE_PREFIX_PATH="/opt/gemfield/opencv4deepvac/;/opt/conda/lib/python3.8/site-packages/torch/" -DCMAKE_INSTALL_PREFIX=../install ..
#cmake(如果基于LibTorch静态库)
cmake -DGARRULOUS_GEMFIELD=ON -DUSE_MKL=ON -DUSE_CUDA=ON -DCMAKE_BUILD_TYPE=Release -DCMAKE_PREFIX_PATH="/opt/gemfield/opencv4deepvac/;/opt/gemfield/libtorch/" -DCMAKE_INSTALL_PREFIX=../install ..
#编译
make -j4
```
- 运行benchmark
```bash
./bin/test_resnet_benchmark cuda:0 <your_torch_script.pt> <a_imagenet_test.jpg>
```

## 2. NA

# 演示
[SYSZUX-FACE](https://github.com/CivilNet/SYSZUX-FACE)基于本项目实现了人脸检测功能。