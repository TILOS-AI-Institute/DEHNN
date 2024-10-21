import os
import shutil
import numpy as np
import pickle
import torch
import torch.nn
from torch_geometric.data import Dataset
from torch_geometric.data import Data

from sklearn.model_selection import train_test_split
from tqdm import tqdm

from collections import defaultdict
from pyg_dataset import NetlistDataset

dataset = NetlistDataset(data_dir="all_designs_netlist_data", load_pe = True, pl = True, processed = True)
