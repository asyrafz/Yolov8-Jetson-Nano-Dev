# Yolov8-Jetson-Nano-Dev
Last Successfully Tested in May 2024
>[!NOTE]
AfterJetpack 4.6 flashed:

>sudo apt-get update

>sudo apt-get install nano

>nano /home/$USER/.bashrc


# check cuda location usr/local
'''
export PATH="/usr/local/cuda-10.2/bin:$PATH"
export LD_LIBRARY_PATH="/usr/local/cuda-10.2/lib64:$LD_LIBRARY_PATH"
'''

'''
dpkg -l | grep cuda
'''
'''
jetson@jetson-desktop:/usr/local/cuda-10.2/bin$ ./nvcc --version
nvcc: NVIDIA (R) Cuda compiler driver
Copyright (c) 2005-2021 NVIDIA Corporation
Built on Sun_Feb_28_22:34:44_PST_2021
Cuda compilation tools, release 10.2, V10.2.300
Build cuda_10.2_r440.TC440_70.29663091_0
'''
# ---step yolov8---
'''
sudo apt update
'''
'''
sudo apt install -y python3.8 python3.8-venv python3.8-dev python3-pip \
libopenmpi-dev libomp-dev libopenblas-dev libblas-dev libeigen3-dev libcublas-dev
'''
'''
git clone https://github.com/ultralytics/ultralytics
cd ultralytics
'''
'''
python3.8 -m venv venv
source venv/bin/activate
'''
'''
pip install -U pip wheel gdown
'''

# pytorch 1.11.0
'''
gdown https://drive.google.com/uc?id=1hs9HM0XJ2LPFghcn7ZMOs5qu5HexPXwM
'''
# torchvision 0.12.0
'''
gdown https://drive.google.com/uc?id=1m0d8ruUY8RvCP9eVjZw4Nc8LAwM8yuGV
python3.8 -m pip install torch-*.whl torchvision-*.whl
'''
'''
pip install .
'''

# Now you can try :scroll:
'''
yolo task=detect mode=predict model=yolov8n.pt source=0 show=True
yolo task=segment mode=predict model=yolov8n-seg.pt source=0 show=True
'''
