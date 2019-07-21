# Tensorflow2 CycleGAN
Converted Jupyter Notebook https://www.tensorflow.org/beta/tutorials/generative/cyclegan to Python 

## Requirements
* tensorflow 2.0.0-beta1
* tensorflow_datasets (horses to zebra dataset)
* tensorflow_examples (pix2pix model)
* jupyter (for visualization in notebooks)
* matplotlib (for visualization in notebooks)


## AWS Install
Instructions below for an AWS p2.xlarge with Deep Learning AMI (Ubuntu 16.04) Version 21.2.

Select CUDA version 10.0
```
sudo rm /usr/local/cuda
sudo ln -s /usr/local/cuda-10.0 /usr/local/cuda
```
Find the appropriate Linux driver version for CUDA 10.0: https://docs.nvidia.com/cuda/cuda-toolkit-release-notes/index.html

Update CUDA driver >= 410.48 (from https://docs.aws.amazon.com/AWSEC2/latest/UserGuide/install-nvidia-driver.html)
```
sudo apt-get update -y
sudo apt-get upgrade -y linux-aws
sudo reboot
sudo apt-get install -y gcc make linux-headers-$(uname -r)
```

```
cat << EOF | sudo tee --append /etc/modprobe.d/blacklist.conf
blacklist vga16fb
blacklist nouveau
blacklist rivafb
blacklist nvidiafb
blacklist rivatv
EOF
```
```
vi /etc/default/grub
```
add following line to grub
```
GRUB_CMDLINE_LINUX="rdblacklist=nouveau"
```

Find correct version of driver https://www.nvidia.com/Download/Find.aspx by selecting Tesla K80, Ubuntu 16.04, CUDA 10.0 and modify version in download path below.

```
sudo update-grub
cd /tmp
wget http://us.download.nvidia.com/tesla/410.104/NVIDIA-Linux-x86_64-410.104.run
sudo /bin/sh ./NVIDIA-Linux-x86_64-410.104.run
```
accept all defaults on install (DKMS will fail if you select it)

```
sudo reboot
```

```
nvidia-smi -q | head
```


## Conda Install
```
conda create -n tf2_p37 python=3.7
conda install -n tf2_p37 pip
source activate tf2_p37
pip install tensorflow-gpu==2.0.0-beta1
pip install --upgrade pip
pip install tensorflow_datasets
pip install jupyter
pip install matplotlib
pip install imageio
pip install -q git+https://github.com/tensorflow/examples.git
```

Test Install
```
import tensorflow as tf
assert tf.test.is_gpu_available()
assert tf.test.is_built_with_cuda()
```

### Download Dataset
when training is run, tensorflow_datasets downloads the cycle_gan dataset (111.45 MiB) to /home/ubuntu/tensorflow_datasets/cycle_gan/horse2zebra/0.1.0
 

## input_params.json Configuration
TODO

## Run Training
```
source activate tf2_p37
cd tf2-cyclegan/
python main.py -c input_params.json
```

## Tensorboard
```
source activate tf2_p37
cd tf2-cyclegan/experiments/
tensorboard --logdir=.
```

## Run Inference on Test Samples
Update trained_checkpoint_dir:
```
vi input_params_predict.json
```

Run inference:
```
source activate tf2_p37
cd tf2-cyclegan/
python predict.py -c input_params_predict.json
```

## Run Tests
TODO