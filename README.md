# DE-HNN: An effective neural model for Circuit Netlist representation

This is the github repository of The AIStats 2024 paper:  
"DE-HNN: An effective neural model for Circuit Netlist representation.
 Z. Luo, T. Hy, P. Tabaghi, D. Koh, M. Defferrard, E. Rezaei, R. Carey, R. Davis, R. Jain and Y. Wang. 27th Intl. Conf. Artificial Intelligence and Statistics (AISTATS), 2024." [arxiv](https://arxiv.org/abs/2404.00477)

## Important

### I have updated this github to fit new version of models and data for easier use. Please read this README carefully for new changes. 

## Environment Setup

We built based on python 3.8.8, CUDA 11.6, Linux Centos 8. Other python version like 3.11, 3.12 should also work.

Please consider using virtual environment like anaconda for easier use. 

The dependencies are recorded in requirements.txt. 

Notice that we put some CUDA-related packages in 
```commandline
cuda_related_package.txt
```
Those packages might need to be manually installed to fit your device's CUDA version. 

If you believe your CUDA version fits the requirements, please run:
```commandline
pip install dgl-cu113 -f https://data.dgl.ai/wheels/repo.html
pip install torch==1.13.1+cu116 torchvision==0.14.1+cu116 torchaudio==0.13.1 --extra-index-url https://download.pytorch.org/whl/cu116
```

Other packages can be installed by run:

```commandline
pip install -r requirements.txt
```

## Memory Usage

We train and test our model on a NVIDIA A100, the model require at least 40 GBs to run the full version (with VNs) and at least 32 GBs to run the smaller version (without VNs).

If you encounter OOM problem, please consider further decrease the size of the model (layers and number of dimensions).

## Data 

For quick usage, reading the data README is strongly recommended but not required. A seperate README for Netlist Data is available here: [Data README](README_DATA.md). 

Raw data, Processed data and Older version data are all available at [link](https://zenodo.org/records/14599896)

If you only want to use the processed data, please download the file **superblue.zip**

After downloading the processed data, please extract and put the data directory to "/de_hnn/data/" directory. You should be able to see a new directory called **superblue** after this.

## How to load the dataset 

After the data is processed (or downloaded), the dataset can be created the instrcutions below:

```commandline
python run_all_data.py
```

A loading process will be initiated immediately. This will create the pytorch-geometric data objects for you to use.

The data objects will be saved at **"superblue/superblue_{design number}/"**

## For running experiments

After dataset are created and saved, next, go to "/dehnn/" directory

In the file **train_all_cross.py**, you can set the hyperparameters and config for the model you want to train, and how to split the dataset. 

Please carefully read through that file and change based on your need (your device, Memory Usage, model type, and more)

And then, run

```commandline
python train_all_cross.py
```

if you want to run with default device, otherwise, you can indicate which cuda device to use, using:

```commandline
CUDA_VISIBLE_DEVICES={your device number} python train_all_cross.py
```

Thank you!
