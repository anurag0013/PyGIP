import copy

import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from torch_geometric.utils import to_dense_adj, dense_to_sparse

from models.nn.backbones import GCN_PyG
from .base import BaseDefense


class OwnerWatermarkingDefense(BaseDefense):

    def __init__(self, dataset, attack_node_fraction=0.3, model_path=None):
        super().__init__(dataset, attack_node_fraction)
        self.model_path = model_path
        self.owner_id = dataset.graph.ndata['feat'].new_tensor([0.1, 0.3, 0.5, 0.7, 0.9])

        in_feats = dataset.graph.ndata['feat'].shape[1]
        num_classes = int(dataset.graph.ndata['label'].max().item()) + 1

        self.generator = TriggerGenerator(in_feats, 64, self.owner_id)
        self.model = GCN_PyG(in_feats, 128, num_classes)

    def defend(self, pyg_graph):
        bi_level_optimization(self.model, self.generator, pyg_graph)
        trigger_data = generate_trigger_graph(pyg_graph, self.generator, self.model)
        metrics = calculate_metrics(self.model, trigger_data)
        return metrics

    def _load_model(self):
        if self.model_path:
            self.model.load_state_dict(torch.load(self.model_path))

    def _train_target_model(self):
        # optional if you split training from watermarking
        pass

    def _train_defense_model(self):
        return self.model

    def _train_surrogate_model(self):
        pass


class TriggerGenerator(nn.Module):
    def __init__(self, in_channels, hidden_channels, owner_id):
        super().__init__()
        self.conv1 = GCNConv(in_channels, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, in_channels)
        self.owner_id = owner_id

    def forward(self, x, edge_index):
        x = F.relu(self.conv1(x, edge_index))
        x = torch.sigmoid(self.conv2(x, edge_index))
        out = x.clone()
        out[:, -5:] = self.owner_id
        return out


def generate_trigger_graph(data, generator, target_model, num_triggers=50):
    with torch.no_grad():
        probs = F.softmax(target_model(data.x, data.edge_index), dim=1)

    selected_nodes = []
    for class_idx in range(probs.size(1)):
        class_nodes = torch.where(data.y == class_idx)[0]
        if len(class_nodes) > 0:
            selected_nodes.append(class_nodes[probs[class_nodes, class_idx].argmax()].item())

    trigger_features = generator(data.x, data.edge_index)
    trigger_nodes = list(range(data.num_nodes, data.num_nodes + num_triggers))
    total_nodes = data.num_nodes + num_triggers

    # Create new dense adjacency matrix
    adj = to_dense_adj(data.edge_index)[0]
    new_adj = torch.zeros((total_nodes, total_nodes), device=adj.device)
    new_adj[:adj.size(0), :adj.size(1)] = adj

    # Connect trigger nodes to selected nodes
    for i, trigger in enumerate(trigger_nodes):
        for node in selected_nodes:
            new_adj[node, trigger] = 1
            new_adj[trigger, node] = 1

    new_data = copy.deepcopy(data)
    new_data.x = torch.cat([data.x, trigger_features[:num_triggers]], dim=0)
    new_data.edge_index = dense_to_sparse(new_adj)[0]
    new_data.y = torch.cat([
        data.y,
        torch.zeros(num_triggers, dtype=torch.long, device=data.y.device)
    ])

    new_data.train_mask = torch.cat([
        data.train_mask,
        torch.zeros(num_triggers, dtype=torch.bool, device=data.x.device)
    ])
    new_data.val_mask = torch.cat([
        data.val_mask,
        torch.zeros(num_triggers, dtype=torch.bool, device=data.x.device)
    ])
    new_data.test_mask = torch.cat([
        data.test_mask,
        torch.zeros(num_triggers, dtype=torch.bool, device=data.x.device)
    ])

    new_data.original_test_mask = data.test_mask.clone()

    # Add trigger info
    new_data.trigger_nodes = trigger_nodes
    new_data.selected_nodes = selected_nodes
    new_data.trigger_mask = torch.zeros(total_nodes, dtype=torch.bool, device=data.x.device)
    new_data.trigger_mask[trigger_nodes] = True

    return new_data


def train_model(model, data, epochs=200, wm_weight=0.2):
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)
    criterion = nn.CrossEntropyLoss()

    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()
        out = model(data.x, data.edge_index)

        main_loss = criterion(out[data.train_mask], data.y[data.train_mask])

        wm_loss = 0
        if hasattr(data, 'trigger_nodes'):
            trigger_mask = data.trigger_mask
            wm_loss = criterion(out[trigger_mask], data.y[trigger_mask])

        loss = (1 - wm_weight) * main_loss + wm_weight * wm_loss
        loss.backward()
        optimizer.step()

    return calculate_metrics(model, data)


def bi_level_optimization(target_model, generator, data, epochs=100, inner_steps=5):
    optimizer_model = torch.optim.Adam(target_model.parameters(), lr=0.01)
    optimizer_gen = torch.optim.Adam(generator.parameters(), lr=0.01)
    criterion = nn.CrossEntropyLoss()

    for epoch in range(epochs):
        for _ in range(inner_steps):
            optimizer_model.zero_grad()
            trigger_data = generate_trigger_graph(data, generator, target_model)

            out_clean = target_model(data.x, data.edge_index)
            out_trigger = target_model(trigger_data.x, trigger_data.edge_index)

            clean_loss = criterion(out_clean[data.train_mask], data.y[data.train_mask])
            trigger_loss = criterion(out_trigger[trigger_data.trigger_mask],
                                     trigger_data.y[trigger_data.trigger_mask])

            total_loss = clean_loss + trigger_loss
            total_loss.backward()
            optimizer_model.step()

        optimizer_gen.zero_grad()
        trigger_data = generate_trigger_graph(data, generator, target_model)

        orig_features = data.x[trigger_data.selected_nodes]
        trigger_features = trigger_data.x[trigger_data.trigger_nodes]
        sim_loss = -F.cosine_similarity(orig_features.unsqueeze(1),
                                        trigger_features.unsqueeze(0), dim=-1).mean()

        out = target_model(trigger_data.x, trigger_data.edge_index)
        trigger_loss = criterion(out[trigger_data.trigger_mask],
                                 trigger_data.y[trigger_data.trigger_mask])

        owner_loss = F.binary_cross_entropy(
            trigger_data.x[trigger_data.trigger_nodes, -5:],
            generator.owner_id.expand(len(trigger_data.trigger_nodes), 5)
        )

        total_gen_loss = 0.4 * sim_loss + 0.4 * trigger_loss + 0.2 * owner_loss
        total_gen_loss.backward()
        optimizer_gen.step()


import torch
from torch_geometric.datasets import Planetoid
from sklearn.metrics import precision_score, recall_score, f1_score
from torch_geometric.transforms import NormalizeFeatures


def load_dataset(name):
    return Planetoid(root=f'/tmp/{name}', name=name, transform=NormalizeFeatures())


def calculate_metrics(model, data):
    model.eval()
    with torch.no_grad():
        out = model(data.x, data.edge_index)
        pred = out.argmax(dim=1)
        true = data.y

        # Handle both original and watermarked data cases
        if hasattr(data, 'original_test_mask'):
            test_mask = data.original_test_mask
            if test_mask.size(0) < pred.size(0):
                pad_len = pred.size(0) - test_mask.size(0)
                test_mask = torch.cat([test_mask, torch.zeros(pad_len, dtype=torch.bool, device=test_mask.device)])
        else:
            test_mask = data.test_mask

        metrics = {
            'accuracy': (pred[test_mask] == true[test_mask]).float().mean().item(),
            'precision': precision_score(true[test_mask].cpu(), pred[test_mask].cpu(), average='macro'),
            'recall': recall_score(true[test_mask].cpu(), pred[test_mask].cpu(), average='macro'),
            'f1': f1_score(true[test_mask].cpu(), pred[test_mask].cpu(), average='macro'),
            'wm_accuracy': None
        }

        if hasattr(data, 'trigger_nodes'):
            wm_mask = torch.zeros(data.x.size(0), dtype=torch.bool, device=data.x.device)  # ✅ fixed here
            wm_mask[data.trigger_nodes] = True
            wm_pred = pred[wm_mask]
            wm_true = true[wm_mask]
            metrics['wm_accuracy'] = (wm_pred == wm_true).float().mean().item() * 100

        return metrics
