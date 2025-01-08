import torch
import torch.nn as nn
import torch.nn.functional as F
from dgl.nn.pytorch import GraphConv

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class Gcn_Net(nn.Module):
    """A simple GCN Network."""
    def __init__(self, feature_number, label_number):
        super(Gcn_Net, self).__init__()
        self.layers = nn.ModuleList()
        self.layers.append(GraphConv(feature_number, 16, activation=F.relu))
        self.layers.append(GraphConv(16, label_number))
        self.dropout = nn.Dropout(p=0.5)
        self.to(device)

    def forward(self, g, features):
        g = g.to(device)
        features = features.to(device)
        x = F.relu(self.layers[0](g, features))
        x = self.layers[1](g, x)
        return x

class Net_shadow(torch.nn.Module):
    """A shadow model GCN."""
    def __init__(self, feature_number, label_number):
        super(Net_shadow, self).__init__()
        self.layer1 = GraphConv(feature_number, 16)
        self.layer2 = GraphConv(16, label_number)

    def forward(self, g, features):
        x = torch.nn.functional.relu(self.layer1(g, features))
        x = self.layer2(g, x)
        return x

class Net_attack(nn.Module):
    """An attack model GCN."""
    def __init__(self, feature_number, label_number):
        super(Net_attack, self).__init__()
        self.layers = nn.ModuleList()
        self.layers.append(GraphConv(feature_number, 16, activation=F.relu))
        self.layers.append(GraphConv(16, label_number))
        self.dropout = nn.Dropout(p=0.5)

    def forward(self, g, features):
        x = F.relu(self.layers[0](g, features))
        x = self.layers[1](g, x)
        return x