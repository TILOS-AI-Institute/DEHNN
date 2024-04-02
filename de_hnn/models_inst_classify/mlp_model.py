import torch
import torch.nn as nn
import torch.nn.functional as F

# Model
class MLP(nn.Module):
    """ A simple Multilayer Perceptron. """

    def __init__(self, input_dim, hidden_dim, output_dim):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim

        self.fc1 = nn.Linear(self.input_dim, self.hidden_dim)
        self.fc2 = nn.Linear(self.hidden_dim, self.hidden_dim)
        self.fc3 = nn.Linear(self.hidden_dim, self.hidden_dim)
        self.fc4 = nn.Linear(self.hidden_dim, self.hidden_dim)
        self.fc5 = nn.Linear(self.hidden_dim, self.output_dim)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        #x = F.dropout(x, p=0.05)
        x = F.relu(self.fc2(x))
        #x = F.dropout(x, p=0.05)
        x = F.relu(self.fc3(x))
        #x = F.dropout(x, p=0.05)
        x = F.relu(self.fc4(x))
        #x = F.dropout(x, p=0.1)
        x = self.fc5(x)
        return F.log_softmax(x, dim=1)

