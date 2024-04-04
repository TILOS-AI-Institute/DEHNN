# DE-HNN: An effective neural model for Circuit Netlist representation

This is the github repository of The AIStats 2024 paper: "DE-HNN: An effective neural model for Circuit Netlist representation" by Zhishang Luo, Truong Son Hy, Puoya Tabaghi, Donghyeon Koh, Michael Defferrard, Elahe Rezaei, Ryan Carey, Rhett Davis, Rajeev Jain, Yusu Wang.

Link to the paper (arxiv preprint): [link](https://arxiv.org/abs/2404.00477)

## Environment Setup

The dependencies are recorded in requirements.txt. You can install the dependencies using pip:

```commandline
pip install -r requirements.txt
```

## How to Download Netlist Data

Raw data is available at [link](https://zenodo.org/records/10795280?token=eyJhbGciOiJIUzUxMiJ9.eyJpZCI6Ijk5NjM2MzZiLTg0ZmUtNDI2My04OTQ3LTljMjA5ZjA3N2Y1OSIsImRhdGEiOnt9LCJyYW5kb20iOiJlYzFmMGJlZTU3MzE1OWMzOTU2MWZkYTE3MzY5ZjRjOCJ9.WifQFExjW1CAW0ahf3e5Qr0OV9c2cw9_RUbOXUsvRbnKlkApNZwVCL_VPRJvAve0MJDC0DDOSx_RLiTvBimr0w). 

After downloading the raw data, please extract and put the raw data directory to "/de_hnn/data/" directory.

After that, please run following command to generate the full processed dataset. The whole process can last for hours.

```commandline
source run_all_python_scripts.sh
```
### Skip Processing Data

If one does not want to process the raw data for any reason but just need the processed data, the full processed data is also available at [link](https://zenodo.org/records/10795280?token=eyJhbGciOiJIUzUxMiJ9.eyJpZCI6Ijk5NjM2MzZiLTg0ZmUtNDI2My04OTQ3LTljMjA5ZjA3N2Y1OSIsImRhdGEiOnt9LCJyYW5kb20iOiJlYzFmMGJlZTU3MzE1OWMzOTU2MWZkYTE3MzY5ZjRjOCJ9.WifQFExjW1CAW0ahf3e5Qr0OV9c2cw9_RUbOXUsvRbnKlkApNZwVCL_VPRJvAve0MJDC0DDOSx_RLiTvBimr0w).

After downloading the processed data, please extract and put the data directory to "/de_hnn/data/" directory.

## How to train 

After the data is processed, the experiments can be run following the instrcutions below. 

### Simple Test

First, enter directory "/de_hnn/experiments/cross_design/".

A simple test on cross-design full-DE-HNN can be run using the following command:

```commandline
source run_simple_test.sh
```
A training process will be initiated immediately. This is a simple test just to test if codes are ready to use. 


A more complete test run with full number of epoches can be run using the following command:

```commandline
source run_full_test.sh
```

The whole process might take 30 minutes to hours depending on devices, which will produce the full results for full-DE-HNN for cross-design.


### For full single-design experiments

All files are ready in directory "/de_gnn/experiments/single_design/".

Please run the commands in this format "source run_all_{model}.sh" depending on which model you want to run. 


### For full cross-design experiments

All files are ready in directory "/de_gnn/experiments/cross_design/".

Please run the commands in this format "source run_all_{model}.sh" depending on which model you want to run.


Thank you!
