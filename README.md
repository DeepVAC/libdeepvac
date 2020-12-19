# libdeepvac
LibTorch project standard.  

这个项目定义了如何在C++项目中使用PyTorch训练的模型。

# 简介

在MLab（云上炼丹师）实验室，我们使用[DeepVAC](https://github.com/DeepVAC/deepvac) 来训练获得新模型，使用本项目来部署模型。

libdeepvac作为一个Linux库，在以下四个方面发挥了价值：
- 向下封装了LibTorch，支持LibTorch静态库和动态库；
- 向上提供Deepvac类，方便用户继承并实现其自定义的模型；
- 在modules目录下，MLab提供了经典网络的C++实现；
- 在utils目录下，MLab提供了网络中常见helper函数的C++实现。

# 模块
### SOTA网络的C++实现
|类名 | 网络 | 作用 |
|-----|------|------|
|SyszuxFaceRetina| RetinaNet | 人脸检测|
|SyszuxOcrPse | PSENet | 文字检测 |
|SyszuxSegEsp | ESPNetV2 | 语义分割 |
|SyszuxClsMobile | MobileNetV3 | 分类 |
|SyszuxDetectYolo | YOLOV5 | 目标检测 |

### helper函数实现
|类名/函数名 | 作用 |
|-----|------|
|AlignFace|人脸对齐|
|nms | 检测框的非极大值抑制|
|PriorBox| 生成目标检测的候选框 |


未来我们会持续在modules、utils目录下提供SOTA网络的C++实现。如果用户（你）需要什么网络的C++实现，可在issues里提交申请。

# 编译平台和目标的支持
- libdeepvac目前仅支持在x86_64 Linux平台上进行编译，未来也不会扩展到更多平台；
- libdeepvac目前仅支持要运行在x86_64 Linux平台上的目标的编译，未来会扩展到Android、ARM Linux。

# 项目依赖
libdeepvac的编译依赖C++17编译器、CMake、opencv、LibTorch。  
你可以直接使用我们提供的Docker image来进行开发：
```bash
docker run -it -h libdeepvac --name libdeepvac gemfield/deepvac:10.2-cudnn7-devel-ubuntu18.04 bash
```
也可以自己手工配置环境依赖，如下所示：

#### C++17
在Ubuntu 20.04上，最新的g++版本为9.3，已经支持C++17。  
在Ubuntu 18.04上，你需要按照如下方式安装支持C++17的编译器：
```bash
apt install software-properties-common
add-apt-repository ppa:ubuntu-toolchain-r/test
apt install gcc-9 g++-9
export CC=/usr/bin/gcc-9
export CXX=/usr/bin/g++-9
```

#### cmake
```bash
apt install cmake
```
  
#### OpenCV;
使用如下的方式来安装预编译的opencv库：
```bash
#install opencv on ubuntu
apt install libopencv-dev
```
如果想要从源码编译，则
```bash
git clone https://github.com/opencv/opencv.git
cd opencv-4.4.0
mkdir build && cd build
cmake .. -DBUILD_LIST=core,imgproc,imgcodecs -DBUILD_SHARED_LIBS=ON/OFF
make -j8 && make install
```

#### LibTorch
支持LibTorch动态库和静态库。因为目前libdeepvac仅支持目标为x86_64 Linux平台目标的编译，所以LibTorch也要下载该平台上的。
- 若想使用LibTorch 1.7.0 动态库，则下载[LibTorch 1.7.0](https://download.pytorch.org/libtorch/cu102/libtorch-cxx11-abi-shared-with-deps-1.7.0.zip)。  
- 若想使用LibTorch 1.6.0 静态库，则下载[LibTorch 1.6.0](https://github.com/CivilNet/libtorch/releases/download/v1.7.0/libtorch_cuda_1.6.0.tar.gz)。
- 若想使用LibTorch 1.7.0 静态库，则下载[LibTorch 1.7.0](https://github.com/CivilNet/libtorch/releases/download/v1.7.0/libtorch_cuda_1.7.0.tar.gz)。


# 编译
如果要开始编译libdeepvac，需要先熟悉如下几个编译开关的作用：
- BUILD_STATIC，libdeepvac支持静态编译和动态编译，使用BUILD_STATIC=ON/OFF 来控制；  
- USE_STATIC_LIBTORCH，libdeepvac支持LibTorch静态库和动态库，使用USE_STATIC_LIBTORCH=ON/OFF 来控制；
- USE_MKL，仅在USE_STATIC_LIBTORCH=ON的情况下生效，是否使用MKL来作为BLAS/LAPACK后端；
- USE_CUDA，仅在USE_STATIC_LIBTORCH=ON的情况下生效，是否使用CUDA。

CMake命令举例如下：

```bash
# create build directory
mkdir build
cd build

#编译libdeepvac动态库，使用LibTorch动态库
cmake -DCMAKE_PREFIX_PATH=/home/gemfield/libtorch/ ..
#编译libdeepvac静态库，使用LibTorch动态库
cmake -DCMAKE_PREFIX_PATH=/home/gemfield/libtorch_cpu/ -DBUILD_STATIC=ON ..
#编译libdeepvac静态库，使用LibTorch静态库
cmake -DCMAKE_PREFIX_PATH=/home/gemfield/libtorch_cuda/ -DBUILD_STATIC=ON -DUSE_STATIC_LIBTORCH=ON -DUSE_MKL=ON -DUSE_CUDA=ON ..


cmake --build . --config Release
```

# 演示
稍后我们会提供一个基于libdeepvac的人脸检测的项目。
