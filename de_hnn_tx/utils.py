import os
import numpy as np
import pickle
import torch
import torch.nn

def compute_accuracy(logits, targets):
    predicted_classes = torch.argmax(logits, dim=1)
    correct_predictions = (predicted_classes.long() == targets.long()).sum().item()
    accuracy = correct_predictions / targets.size(0)
    return accuracy