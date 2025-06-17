import time
import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
import dgl
import numpy as np
import torch
import torch.nn.functional as F
from torch_geometric.utils import k_hop_subgraph, dense_to_sparse
from tqdm import tqdm

from models.attack.base import BaseAttack
from utils.metrics import GraphNeuralNetworkMetric
from models.nn import GCN

# Use device from base class
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class AdversarialModelExtraction(BaseAttack):
    def __init__(self, dataset, attack_node_fraction, model_path=None):
        super().__init__(dataset, attack_node_fraction, model_path)

    def _load_model(self, model_path):
        """
        Load a pre-trained model.
        """
        from models.nn import GCN
        
        # Create the model
        self.net1 = GCN(self.feature_number, self.label_number).to(device)
        
        # Load the saved state dict
        self.net1.load_state_dict(torch.load(model_path, map_location=device))
        
        # Set to evaluation mode
        self.net1.eval()

    # Define a local to_cpu method to avoid inheritance issues
    def _to_cpu(self, tensor):
        """
        Safely move tensor to CPU for NumPy operations
        """
        if tensor.is_cuda:
            return tensor.cpu()
        return tensor

    def attack(self):
        g = self.graph.clone()
        # Move adjacency matrix to CPU for NumPy operations
        g_matrix = np.asmatrix(self._to_cpu(g.adjacency_matrix().to_dense()).numpy())
        edge_index = np.array(np.nonzero(g_matrix))
        edge_index = torch.tensor(edge_index, dtype=torch.long)
        
        # Select a center node with certain size
        while True:
            node_index = torch.randint(0, self.node_number, (1,)).item()
            # print("node_index=",node_index)
            sub_node_index, sub_edge_index, _, _ = k_hop_subgraph(node_index, 2, edge_index, relabel_nodes=True,
                                                                  num_nodes=self.node_number)
            if 45 <= sub_node_index.size(0) <= 50:
                As = torch.zeros((sub_node_index.size(0), sub_node_index.size(0)))
                As[sub_edge_index[0], sub_edge_index[1]] = 1
                print("sub_node_index=",sub_node_index.size(0))
                # Ensure moved to CPU
                Xs = self._to_cpu(self.features[sub_node_index])
                break

        # Construct the prior distribution
        Fd = []
        Md = []
        for label in range(self.label_number):
            # Ensure moved to CPU before converting to NumPy
            features_cpu = self._to_cpu(self.features)
            labels_cpu = self._to_cpu(self.labels)
            class_nodes = features_cpu[labels_cpu == label].numpy()

            feature_counts = class_nodes.sum(axis=0)
            feature_distribution = feature_counts / feature_counts.sum()
            Fd.append(feature_distribution)

            num_features_per_node = class_nodes.sum(axis=1)
            feature_count_distribution = np.bincount(num_features_per_node.astype(int), minlength=self.feature_number)
            Md.append(feature_count_distribution / feature_count_distribution.sum())

        SA = [As]
        SX = [Xs]
        
        # Query the target model
        self.net1.eval()
        with torch.no_grad():
            logits_query = self.net1(g, self.features)
            _, labels_query = torch.max(logits_query, dim=1)

        src, dst = As.nonzero(as_tuple=True)
        initial_num_nodes = Xs.shape[0]
        initial_graph = dgl.graph((src, dst), num_nodes=initial_num_nodes).to(device)
        initial_graph.ndata['feat'] = Xs.to(device)

        self.net1.eval()
        with torch.no_grad():
            initial_query = self.net1(initial_graph, initial_graph.ndata['feat'])
            _, initial_label = torch.max(initial_query, dim=1)

        SL = self._to_cpu(initial_label).tolist()
        samples_per_class = 10
        n = samples_per_class

        for i in range(n):
            # For each class, generate and store a new sampled subgraph
            for c in range(self.label_number):
                num_nodes = As.shape[0]
                Ac = torch.ones((num_nodes, num_nodes))
                Xc = torch.zeros(num_nodes, len(Fd[c]))
                for j in range(num_nodes):  # Use j to avoid conflict with outer loop variable i
                    m = np.random.choice(np.arange(len(Md[c])), p=Md[c])
                    features_idx = np.random.choice(len(Fd[c]), size=int(m), replace=False, p=Fd[c])
                    Xc[j, features_idx] = 1
                SA.append(Ac)
                SX.append(Xc)

                src, dst = Ac.nonzero(as_tuple=True)
                subgraph = dgl.graph((src, dst), num_nodes=num_nodes).to(device)
                subgraph.ndata['feat'] = Xc.to(device)

                self.net1.eval()
                with torch.no_grad():
                    api_query = self.net1(subgraph, subgraph.ndata['feat'])
                    _, label_query = torch.max(api_query, dim=1)

                SL.extend(self._to_cpu(label_query).tolist())

        AG_list = [dense_to_sparse(torch.tensor(a))[0] for a in SA]
        XG = torch.vstack([torch.tensor(x) for x in SX])

        SL = torch.tensor(SL, dtype=torch.long)

        # Filter valid labels and trim
        valid_mask = SL >= 0
        SL = SL[valid_mask]
        SL = SL[:XG.shape[0]]

        # Calculate nodes per subgraph
        num_nodes = XG.shape[0] // len(AG_list) if len(AG_list) > 0 else 0

        # Combine edge indices from all subgraphs, adjusting node indices to avoid overlap
        AG_combined = torch.cat([edge_index + i * num_nodes for i, edge_index in enumerate(AG_list)], dim=1)

        src, dst = AG_combined[0], AG_combined[1]
        num_total_nodes = XG.shape[0]
        sub_g = dgl.graph((src, dst), num_nodes=num_total_nodes).to(device)
        sub_g.ndata['feat'] = XG.to(device)

        # Create and train the extracted model
        net6 = GCN(XG.shape[1], self.label_number).to(device)
        optimizer = torch.optim.Adam(net6.parameters(), lr=0.01, weight_decay=5e-4)

        dur = []

        print("=========Model Extracting==========================")
        best_performance_metrics = GraphNeuralNetworkMetric()
        
        for epoch in tqdm(range(200)):
            if epoch >= 3:
                t0 = time.time()

            net6.train()
            logits = net6(sub_g, sub_g.ndata['feat'])
            out = torch.log_softmax(logits, dim=1)
            loss = F.nll_loss(out, SL.to(device))

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if epoch >= 3:
                dur.append(time.time() - t0)
                
            # Switch to evaluation mode
            net6.eval()
            with torch.no_grad():
                focus_gnn_metrics = GraphNeuralNetworkMetric(
                    0, 0, net6, g, self.features, self.test_mask, self.labels, self._to_cpu(labels_query))
                focus_gnn_metrics.evaluate()

                best_performance_metrics.fidelity = max(
                    best_performance_metrics.fidelity, focus_gnn_metrics.fidelity)
                best_performance_metrics.accuracy = max(
                    best_performance_metrics.accuracy, focus_gnn_metrics.accuracy)

        print("========================Final results:=========================================")
        print(best_performance_metrics)
        
        self.net2 = net6

        return best_performance_metrics, net6