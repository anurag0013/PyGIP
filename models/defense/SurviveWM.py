import time

import dgl
import networkx as nx
import torch
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score
from torch_geometric.data import Data
from tqdm import tqdm

from models.defense.base import BaseDefense
from models.nn import GCN
from utils.metrics import GraphNeuralNetworkMetric

# Use device from base class
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class SurviveWM(BaseDefense):
    def __init__(self, dataset, attack_node_fraction, model_path=None):
        super().__init__(dataset, attack_node_fraction)
        assert dataset.api_type == 'dgl', 'only support dgl dataset'
        # load graph data
        self.dataset = dataset
        self.graph_dataset = dataset.graph_data
        self.graph_data = dataset.graph_data
        self.model_path = model_path
        self.graph = self.graph_data
        self.features = dataset.graph_data.ndata['feat']
        self.labels = dataset.graph_data.ndata['label']
        self.train_mask = dataset.graph_data.ndata['train_mask']
        self.test_mask = dataset.graph_data.ndata['test_mask']

        # load meta data
        self.feature_number = dataset.num_features
        self.label_number = dataset.num_classes

        # params
        self.attack_node_fraction = attack_node_fraction

    def _load_model(self):
        """
        Load a pre-trained model.
        """
        from models.nn import GCN
        assert self.model_path is not None, "Please provide a pre-trained model"

        # Create the model
        self.net1 = GCN(self.feature_number, self.label_number).to(device)

        # Load the saved state dict
        self.net1.load_state_dict(torch.load(self.model_path, map_location=device))

        # Set to evaluation mode
        self.net1.eval()

    def _to_cpu(self, tensor):
        """
        Safely move tensor to CPU for NumPy operations
        """
        if tensor.is_cuda:
            return tensor.cpu()
        return tensor

    # === Soft Nearest Neighbor Loss ===
    def snn_loss(self, x, y, T=0.5):
        x = F.normalize(x, p=2, dim=1)
        dist_matrix = torch.cdist(x, x, p=2) ** 2
        eye = torch.eye(len(x), device=x.device).bool()
        sim = torch.exp(-dist_matrix / T)
        mask_same = y.unsqueeze(1) == y.unsqueeze(0)
        sim = sim.masked_fill(eye, 0)
        denom = sim.sum(1)
        nom = (sim * mask_same.float()).sum(1)
        loss = -torch.log(nom / (denom + 1e-10) + 1e-10).mean()
        return loss

    # === Trigger Graph Generator ===
    def generate_key_graph(self, num_nodes=10, edge_prob=0.3):
        trigger = nx.erdos_renyi_graph(num_nodes, edge_prob)
        edge_index = torch.tensor(list(trigger.edges), dtype=torch.long).t().contiguous()
        if edge_index.numel() == 0:
            edge_index = torch.empty((2, 0), dtype=torch.long)

        x = torch.randn((num_nodes, self.feature_number)) * 0.1
        label = torch.randint(0, self.label_number, (num_nodes,))
        return Data(x=x, edge_index=edge_index, y=label)

    # === Combine base and trigger ===
    def combine_with_trigger(self, base_graph, base_features, base_labels, trigger_data):
        # Convert DGL graph to edge_index format
        src, dst = base_graph.edges()
        base_edge_index = torch.stack([src, dst], dim=0)

        x = torch.cat([base_features, trigger_data.x], dim=0)
        edge_index = torch.cat([base_edge_index, trigger_data.edge_index + base_features.size(0)], dim=1)
        y = torch.cat([base_labels, trigger_data.y], dim=0)

        # Create DGL graph from combined data
        src_combined, dst_combined = edge_index[0], edge_index[1]
        combined_graph = dgl.graph((src_combined, dst_combined), num_nodes=x.size(0)).to(device)

        # **FIX: Add self-loops to handle zero in-degree nodes**
        combined_graph = dgl.add_self_loop(combined_graph)

        combined_graph.ndata['feat'] = x.to(device)

        return combined_graph, y.to(device)

    def train_with_snnl(self, model, graph, features, labels, train_mask, optimizer, T=0.5, alpha=0.1):
        model.train()
        optimizer.zero_grad()
        out = model(graph, features)
        loss_nll = F.nll_loss(F.log_softmax(out, dim=1)[train_mask], labels[train_mask])
        snnl = self.snn_loss(out[train_mask], labels[train_mask], T=T)
        loss = loss_nll - alpha * snnl
        loss.backward()
        optimizer.step()
        return loss.item()

    @torch.no_grad()
    def verify_watermark(self, model, trigger_graph, trigger_labels):
        model.eval()
        out = model(trigger_graph, trigger_graph.ndata['feat'])
        pred = out.argmax(dim=1)
        return (pred == trigger_labels).float().mean().item()

    def compute_metrics(self, y_true, y_pred, y_score=None):
        return {
            'accuracy': accuracy_score(y_true, y_pred),
            'f1': f1_score(y_true, y_pred, average='macro'),
            'precision': precision_score(y_true, y_pred, average='macro'),
            'recall': recall_score(y_true, y_pred, average='macro'),
            'auroc': roc_auc_score(y_true, y_score, multi_class='ovo') if y_score is not None else None
        }

    def defend(self):
        print("=========SurviveWM Attack==========================")

        # Generate trigger graph
        trigger_data = self.generate_key_graph().to(device)

        # Combine base graph with trigger
        combined_graph, combined_labels = self.combine_with_trigger(
            self.graph, self.features, self.labels, trigger_data)

        # Create train mask for combined data (include trigger nodes in training)
        base_train_mask = self.train_mask
        trigger_train_mask = torch.ones(trigger_data.num_nodes, dtype=torch.bool, device=device)
        combined_train_mask = torch.cat([base_train_mask, trigger_train_mask])

        # Create test mask for combined data (exclude trigger nodes from testing)
        base_test_mask = self.test_mask
        trigger_test_mask = torch.zeros(trigger_data.num_nodes, dtype=torch.bool, device=device)
        combined_test_mask = torch.cat([base_test_mask, trigger_test_mask])

        # Create watermarked model
        watermarked_model = GCN(self.feature_number, self.label_number).to(device)
        optimizer = torch.optim.Adam(watermarked_model.parameters(), lr=0.01, weight_decay=5e-4)

        # Create trigger graph for watermark verification
        trigger_src, trigger_dst = trigger_data.edge_index[0], trigger_data.edge_index[1]
        trigger_graph = dgl.graph((trigger_src, trigger_dst), num_nodes=trigger_data.num_nodes).to(device)

        # **FIX: Add self-loops to trigger graph as well**
        trigger_graph = dgl.add_self_loop(trigger_graph)

        trigger_graph.ndata['feat'] = trigger_data.x.to(device)

        dur = []
        best_performance_metrics = GraphNeuralNetworkMetric()

        print("Training watermarked model...")
        for epoch in tqdm(range(200)):
            if epoch >= 3:
                t0 = time.time()

            # Train with SNNL
            loss = self.train_with_snnl(
                watermarked_model, combined_graph, combined_graph.ndata['feat'],
                combined_labels, combined_train_mask, optimizer)

            if epoch >= 3:
                dur.append(time.time() - t0)

            # Evaluate periodically
            if epoch % 20 == 0:
                watermarked_model.eval()
                with torch.no_grad():
                    # Test on original graph (ensure it has self-loops)
                    test_graph = dgl.add_self_loop(self.graph)

                    logits = watermarked_model(test_graph, self.features)
                    pred = logits.argmax(dim=1)
                    test_acc = (pred[self.test_mask] == self.labels[self.test_mask]).float().mean()

                    # Verify watermark
                    wm_acc = self.verify_watermark(watermarked_model, trigger_graph, trigger_data.y)

                    print(f"Epoch {epoch}, Test Acc: {test_acc.item():.4f}, Watermark Acc: {wm_acc:.4f}")

        # Final evaluation
        watermarked_model.eval()
        with torch.no_grad():
            # Evaluate on test set (ensure graph has self-loops)
            test_graph = dgl.add_self_loop(self.graph)

            logits = watermarked_model(test_graph, self.features)
            pred = logits.argmax(dim=1)
            probs = F.softmax(logits, dim=1)

            test_metrics = self.compute_metrics(
                self._to_cpu(self.labels[self.test_mask]).numpy(),
                self._to_cpu(pred[self.test_mask]).numpy(),
                self._to_cpu(probs[self.test_mask]).numpy()
            )

            # Verify watermark
            wm_acc = self.verify_watermark(watermarked_model, trigger_graph, trigger_data.y)

            # Create custom metrics object
            final_metrics = GraphNeuralNetworkMetric()
            final_metrics.accuracy = test_metrics['accuracy']
            final_metrics.fidelity = wm_acc  # Use watermark accuracy as fidelity measure

            print("========================Final results:=========================================")
            print(f"Test Accuracy: {test_metrics['accuracy']:.4f}")
            print(f"Test F1: {test_metrics['f1']:.4f}")
            print(f"Test Precision: {test_metrics['precision']:.4f}")
            print(f"Test Recall: {test_metrics['recall']:.4f}")
            print(f"Watermark Accuracy: {wm_acc:.4f}")
            print(final_metrics)

        self.net2 = watermarked_model

        return final_metrics, watermarked_model
