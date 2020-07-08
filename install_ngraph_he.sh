#!  /bin/bash

# This file is used only to download all the dependencies for HE_Transformer and install it.

sudo apt-get update && apt-get install -y \
    python3-pip virtualenv \
    python3-numpy python3-dev python3-wheel \
    git \
    unzip wget \
    sudo \
    bash-completion \
    build-essential cmake \
    software-properties-common \
    git \
    wget patch diffutils libtinfo-dev \
    autoconf libtool \
    doxygen graphviz \
    yapf3 python3-yapf

# Install clang-9
sudo wget -O - https://apt.llvm.org/llvm-snapshot.gpg.key | sudo apt-key add -
sudo apt-add-repository "deb http://apt.llvm.org/bionic/ llvm-toolchain-bionic-9 main"
sudo apt-get update && apt install -y clang-9 clang-tidy-9 clang-format-9

sudo apt-get clean autoclean && \
        apt-get autoremove -y

# For ngraph-tf integration testing
sudo pip3 install --upgrade pip setuptools virtualenv==16.1.0

# SEAL requires newer version of CMake
sudo pip3 install cmake --upgrade

sudo cmake --version
sudo make --version
sudo gcc --version
sudo clang++-9 --version
sudo c++ --version
sudo python3 --version
sudo virtualenv --version

# Get bazel for ng-tf
sudo wget https://github.com/bazelbuild/bazel/releases/download/0.25.2/bazel-0.25.2-installer-linux-x86_64.sh
sudo chmod +x ./bazel-0.25.2-installer-linux-x86_64.sh
sudo bash ./bazel-0.25.2-installer-linux-x86_64.sh

# Build HE-Transformer
git clone https://github.com/IntelAI/he-transformer.git
cd he-transformer
export HE_TRANSFORMER=$(pwd)
mkdir build
cd $HE_TRANSFORMER/build
sudo cmake .. -DCMAKE_CXX_COMPILER=clang++-6.0

sudo make install

cd $HE_TRANSFORMER/build
source external/venv-tf-py3/bin/activate
sudo make install python_client
sudo pip install python/dist/pyhe_client-*.whl
python3 -c "import pyhe_client"

# Run C++ unit tests for HE-Transformer
cd $HE_TRANSFORMER/build
# To run single HE_SEAL unit-test
./test/unit-test --gtest_filter="HE_SEAL.add_2_3_cipher_plain_real_unpacked_unpacked"
# To run all C++ unit-tests
./test/unit-test