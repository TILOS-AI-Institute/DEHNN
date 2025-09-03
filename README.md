# DE-HNN: An effective neural model for Circuit Netlist representation

This is the github repository of The AIStats 2024 paper:  
"DE-HNN: An effective neural model for Circuit Netlist representation.
 Z. Luo, T. Hy, P. Tabaghi, D. Koh, M. Defferrard, E. Rezaei, R. Carey, R. Davis, R. Jain and Y. Wang. 27th Intl. Conf. Artificial Intelligence and Statistics (AISTATS), 2024." [arxiv](https://arxiv.org/abs/2404.00477)

## Important

This repository has been updated to fit new version of models and data for easier use. Please read this README carefully for the latest changes.

## Environment Setup

We built based on python 3.8.8, CUDA 11.6, Linux Centos 8. Other python version like 3.11, 3.12 should also work.

Please consider using virtual environment like anaconda for easier use.

The dependencies are recorded in requirements.txt.

Notice that we put some CUDA-related packages in:
```commandline
cuda_related_package.txt
```
Those packages might need to be manually installed to fit your device's CUDA version.

If you believe your CUDA version fits the requirements, please run:
```commandline
pip install dgl-cu113 -f https://data.dgl.ai/wheels/repo.html
pip install torch==1.13.1+cu116 torchvision==0.14.1+cu116 torchaudio==0.13.1 --extra-index-url https://download.pytorch.org/whl/cu116
```

Other packages can be installed by running:
```commandline
pip install -r requirements.txt
```

## Memory Usage

We train and test our model on a NVIDIA A100, the model require at least 40 GBs to run the full version (with VNs) and at least 32 GBs to run the smaller version (without VNs).

If you encounter OOM problem, please consider further decrease the size of the model (layers and number of dimensions).

## Dataset Information

### Raw Data
Digital Integrated Circuit Graph Data: SKY130-HS RocketTile Data

2024-01-15 by Donghyeon Koh and W. Rhett Davis, NC State University

The netlist dataset consists of 12 of the Superblue circuits from (Viswanathan et al., 2011, 2012), including Superblue1,2,3,5,6,7,9,11,14,16,18 and 19. The size of these netlists range from 400K to 1.3M nodes, with similar number of nets. More details of designs can be found in paper and appendix.

These netlist files were generated with physical design of the Rocket-Chip generator [link](https://github.com/chipsalliance/rocket-chip) for the Skywater 130nm process and High-Speed standard-cell library (sky130hs). The default configuration of the Rocket-Chip was used, and physical design was performed for the RocketTile module. Dummy memories were generated with a similar interface to the OpenRAM single-port SRAM.

The database-units-to-user-units (DBUtoUU) conversion factor for this dataset is 1000. Integer dimensions should be divided by this factor to get real dimensions in microns.

The file settings.csv contains the following settings for each variant:
- CORE_UTILIZATION - initial ratio of cell area to core area
- MAX_ROUTING_LAYER - maximum allowed layer for routing (complete list of layers is in counter_congestion.npz 'layerList')
- CLK_PER - clock period constraint (i.e. target) in units of ns
- MAX_CLK_TRANS - maximum allowed transition time for a clock node, in units of ns. These are currently set at 500 ns for all variants, which is effectively unconstrained.
- CLK_UNCERTAINTY - clock uncertainty (currently 0.2 ns for all variants)
- FLOW_STAGE - Design flow stage at which the data was generated

There are 6 other settings relating to the layout of the power distribution network, but these are fixed for all variants and can be ignored for now: HSTRAP_LAYER, HSTRAP_WIDTH, HSTRAP_PITCH, VSTRAP_LAYER, VSTRAP_WIDTH, and VSTRAP_PITCH

Each flow-stage has time-stamps for the beginning and end of execution, labeled "begin_time" and "end_time". These time-stamps were created with "date +%s" and give seconds since 1970-01-01 UTC.

There are 8 additional outcomes/labels available for each flow-stage after "init_design", each calculated at the end of the stage:
- wnhs - worst negative hold-slack for any register input, in units of ns
- tnhs - total negative hold-slack, summed for all register inputs, in units of ns
- nhve - number of hold-violation enpoints, i.e. the number of register inputs with negative hold slack
- ntv - number of total violations of a maximum transition-time setting on any net
- critpath - critical-path delay in units of ns
- max_clk_trans_out - maximum transition time for any clock node, in units of ns
- area_cell - cell area in units of square microns
- core_utilization_out - ratio of cell area to core area

### Processed Data
We used PyTorch-geometric (pyg) (Fey and Lenssen, 2019) to construct the dataset and data objects. Depending on the models is a Graph Neural Network or a (directed) Hypergraph Neural Network, each netlist circuit from the Raw Data will be represented as a bipartite-graph or (directed) hypergraph using pyg.

**Features:**
- Cell/Node Features:
  * Type (int): Master library cell ID (array index).
  * Orient (int): Orientation of a cell.
  * Width, Height (float): Width and height of a cell.
  * Cell Degree (int): The degree of a cell.
  * Degree Distribution (list[int]): Degree distribution of a local neighborhood.
  * Laplacian Eigenvector (list[float]): Top-10 Laplacian Eigenvector.
  * PD (list[float]): Persistent diagram features.
- Net/(Hyper)edge Feature:
  * Net Degree (int): The degree of a net.

**Targets:**
- Net-based Wirelength Regression: Half-perimeter wirelength (HPWL) as a common estimate of wirelength.
- Net-based Demand Regression: Demand of each net, congestion happens when demand exceeds capacity.
- Cell-based Congestion Classification: Similar to (Yang et al., 2022) and (Wang et al., 2022), we classify the cell-based congestion values (computed as the ratio of cell demand/cell capacity) into (a) [0,0.9], not-congested; and (b) [0.9, inf]; congested.

## Data Download and Setup

Raw data, Processed data and Older version data are all available at [link](https://zenodo.org/records/14599896)

If you only want to use the processed data, please download the file **superblue.zip**

After downloading the processed data, please extract and put the data directory to "/de_hnn/data/" directory. You should be able to see a new directory called **superblue** after this.

In folder "superblue/" which can be downloaded at [link](https://zenodo.org/records/14599896), all the files corresponding to each design are expressed as "{design_number}/{file_name}.pkl"

Below is a file description for each design/netlist (there is only one netlist for each design):
- bipartite.pkl: The connectivity information between cells and nets of the bipartite graph representation of a netlist.
- degree.pkl: The degrees information of cells and nets.
- eigen.10.pkl: The top-10 eigenvectors and eigenvalues.
- global_information.pkl: The global information of a netlist.
- metis_part_dict.pkl: The Metis (Karypis and Kumar, 1998) [link](https://github.com/KarypisLab/METIS) based partition information.
- net_demand_capacity.pkl: The demands and capacity information of each net.
- net_features.pkl: The features of each net.
- net_hpwl.pkl: The Half-perimeter wirelength (HPWL) for each net.
- nn_conn.pkl: The connectivity file prepared for NetlistGNN (Yang et al., 2022). [link](https://github.com/PKUterran/NetlistGNN)
- node_features.pkl: The features of each node.
- targets.pkl: The processed targets or labels of each cell/net to predict.
- node_neighbors/{idx}.node_neighbor_features.pkl: The neighborhood features for each cell/node.

Notice that there are another several files, which are not related to the main text in our paper, but is used in our appendix:
- pl_fix_part_dict.pkl: The fixed-size bounding box based partition information (when placement info available).
- pl_part_dict.pkl: The relative-size bounding box (the number of boxes is same for all netlist) based partition information (when placement info available).
- star.pkl: The "star" graph representation where only cells and the connectivities between cells are included. Please refer to appendix for more details.
- star_part_dict: The Metis based partition, but on star graph representation of each netlist.

## How to load the dataset

After the data is processed (or downloaded), the dataset can be created with the instructions below:

```commandline
python run_all_data.py
```

A loading process will be initiated immediately. This will create the pytorch-geometric data objects for you to use.

The data objects will be saved at **"superblue/superblue_{design number}/"**

## Model Architecture

PyTorch Geometric (PyG) models are in ```gnn.py``` with the graph convolution implemented in ```graph_conv.py```. We can also use the position encodings from GraphGPS (check ```position_enc/``` for more options).

The baseline of Linear Transformer + LapPE (Laplacian position encoding) is in ```linear_transformer_lappe.py```.

Directory gnn_to_exchange contains two other implementations, one is transfer directed graph to undirected graph, another one implemented directed gcn with two separate message passing for (source to target) and (target to source). Current reported results coming from the directed gcn version.

## Running Experiments

After dataset are created and saved, next, go to "/de_hnn/" directory

In the file **train_all_cross.py**, you can set the hyperparameters and config for the model you want to train, and how to split the dataset.

Please carefully read through that file and change based on your need (your device, Memory Usage, model type, and more)

And then, run:

```commandline
python train_all_cross.py
```

if you want to run with default device, otherwise, you can indicate which cuda device to use, using:

```commandline
CUDA_VISIBLE_DEVICES={your device number} python train_all_cross.py
```

## File Structure

- `de_hnn/` - Main model implementation directory
  - `data/` - Data processing and loading scripts
  - `models/` - Model implementations
  - `train_all_cross.py` - Main training script
- `older_version/` - Contains older model implementations for reference
- `requirements.txt` - Python dependencies
- `cuda_related_package.txt` - CUDA-specific package requirements

Thank you!