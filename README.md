# libdeepvac
LibTorch project standard.  
这个项目定义了如何在C++项目中使用PyTorch训练的模型。

## 项目依赖
- libtorch；
- OpenCV;
- C++17;
- cmake;

```bash
#install opencv on ubuntu
apt install libopencv-dev

#Ubuntu 20.04上，最新的g++版本为9.3，已经支持C++17；
#Ubuntu 18.04上，你需要按照如下方式安装支持C++17的编译器：
apt install software-properties-common
add-apt-repository ppa:ubuntu-toolchain-r/test
apt install gcc-9 g++-9
export CC=/usr/bin/gcc-9
export CXX=/usr/bin/g++-9
```

## 若打算从源码编译opencv，方法如下
```bash
git clone https://github.com/opencv/opencv.git
cd opencv-4.4.0
mkdir build && cd build
cmake .. -DBUILD_LIST=core,imgproc,imgcodes -DBUILD_SHARED_LIBS=ON
make -j8 && make install
```

## 编译
```bash
# create build directory
mkdir build
cd build
cmake -DCMAKE_PREFIX_PATH=/home/gemfield/libtorch/ ..
cmake --build . --config Release
```
