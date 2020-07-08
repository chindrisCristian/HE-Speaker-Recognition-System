#! /bin/bash

# Install HE-Transformer

./install_ngraph_he.sh

# Install python3 tensorflow, keras

sudo pip3 install tensorflow
sudo pip3 install keras

# Train server's neural network used for classification

python3 Train.py --epochs=20

# Activate the virtual environment for running ngraph_bridge
source $HE_TRANSFORMER/build/external/venv-tf-py3/bin/activate