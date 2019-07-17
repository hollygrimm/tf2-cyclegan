# Tensorflow2 CycleGAN


## Requirements
* tensorflow 2.0.0-beta1


## AWS Install


## Manual Install
```
conda create -n tf2_p37 python=3.7
conda install -n tf2_p37 pip
source activate tf2_p37
pip install tensorflow-gpu==2.0.0-beta1
pip install --upgrade pip
pip install tensorflow_datasets
```

Test Install
```
import tensorflow as tf
assert tf.test.is_gpu_available()
assert tf.test.is_built_with_cuda()
```

### Download Dataset


## Label Data with Attributes

## input_params.json Configuration


## Run Training
```
source activate tf2_p37
cd tf2-cyclegan/
python main.py -c input_params.json
```

## Tensorboard


## Run Inference on Validation Samples


## Run Tests
