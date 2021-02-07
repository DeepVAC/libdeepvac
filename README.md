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
|SyszuxClsResnet | ResNet50 | 分类 |

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
最简便、高效的方式就是使用我们提供的DeepVAC开发时[Docker镜像](https://github.com/DeepVAC/deepvac#2-%E7%8E%AF%E5%A2%83%E5%87%86%E5%A4%87)。

- 该镜像内置的LibTorch版本为：[LibTorch4DeepVAC (x86-64 Linux) 1.8.0](https://github.com/CivilNet/libtorch/releases/download/1.8.0/libtorch.tar.gz)；
- 你也可以使用[PyTorch官方](https://pytorch.org/)LibTorch版本替换内置的LibTorch版本；
- 未来几个月，该镜像还将包含LibTorch4DeepVAC (Android) 1.8.0；


# 编译
如果要开始编译libdeepvac，需要先熟悉如下几个编译开关的作用：
- BUILD_STATIC，libdeepvac支持静态编译和动态编译，使用BUILD_STATIC=ON/OFF 来控制；  
- USE_STATIC_LIBTORCH，libdeepvac支持LibTorch静态库和动态库，使用USE_STATIC_LIBTORCH=ON/OFF 来控制；
- USE_MKL，仅在USE_STATIC_LIBTORCH=ON的情况下生效，是否使用MKL来作为BLAS/LAPACK后端；
- USE_CUDA，仅在USE_STATIC_LIBTORCH=ON的情况下生效，是否使用CUDA。

CMake命令如下：

```bash
# create build directory
mkdir build
cd build

cmake -DBUILD_STATIC=ON  -DUSE_STATIC_LIBTORCH=ON -DUSE_MKL=ON -DUSE_CUDA=ON -DCMAKE_BUILD_TYPE=Release -DCMAKE_PREFIX_PATH="/gemfield/libtorch;/gemfield/opencv4deepvac/" -DCMAKE_INSTALL_PREFIX=../install .. 

cmake --build . --config Release
make install
```

# 演示
[SYSZUX-FACE](https://github.com/CivilNet/SYSZUX-FACE)基于本项目实现了人脸检测功能。

