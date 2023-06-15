# Description
**DeltaBoost** is a machine learning model based on gradient boosting decision trees (GBDT) that supports efficient machine unlearning. The implementation of DeltaBoost is based on standalone FedTree.
# Getting Started
The documentation of FedTree can be found [here](https://fedtree.readthedocs.io/en/latest/index.html).
## Prerequisites
* [CMake](https://cmake.org/) 3.15 or above
* [NTL](https://libntl.org/) library
* [Boost](https://www.boost.org/) 1.75 or above

You can follow the following commands to install NTL library.

```
wget https://libntl.org/ntl-11.4.4.tar.gz
tar -xvf ntl-11.4.4.tar.gz
cd ntl-11.4.4/src
./configure
make
make check
sudo make install
```


If you install the NTL library at another location, please also modify the CMakeList files of FedTree accordingly (line 64 of CMakeLists.txt).
## Install submodules
```
git submodule init thrust
git submodule update
```

## Build on Linux

```bash
mkdir build && cd build
cmake .. -DDISTRIBUTED=OFF
make -j
```

## Training Example
```bash
# under 'FedTree' directory
./build/bin/FedTree-train ./conf/tree1/cadata_1e-02.conf
```

# Other information
FedTree is built based on [ThunderGBM](https://github.com/Xtra-Computing/thundergbm), which is a fast GBDTs and Radom Forests training system on GPUs.
