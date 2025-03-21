import time

import dgl
import numpy as np
import torch
import torch.nn.functional as F
from torch_geometric.utils import k_hop_subgraph, dense_to_sparse
from tqdm import tqdm

from models.attack.base import BaseAttack
from utils.metrics import GraphNeuralNetworkMetric
from models.nn import GCN


class AdversarialModelExtraction(BaseAttack):
    def __init__(self, dataset, attack_node_fraction, model_path=None):
        super().__init__(dataset, attack_node_fraction, model_path)

    def attack(self):
        g = self.graph.clone()
        g_matrix = np.asmatrix(g.adjacency_matrix().to_dense())
        edge_index = np.array(np.nonzero(g_matrix))
        edge_index = torch.tensor(edge_index, dtype=torch.long)
        # select a center node with certain size
        while True:
            node_index = torch.randint(0, self.node_number, (1,)).item()
            sub_node_index, sub_edge_index, _, _ = k_hop_subgraph(node_index, 2, edge_index, relabel_nodes=True,
                                                                  num_nodes=self.node_number)
            if 10 <= sub_node_index.size(0) <= 150:
                As = torch.zeros((sub_node_index.size(0), sub_node_index.size(0)))
                As[sub_edge_index[0], sub_edge_index[1]] = 1

                Xs = self.features[sub_node_index]
                break

        # construct the prior distribution
        Fd = []
        Md = []
        for label in range(self.label_number):
            class_nodes = self.features[self.labels == label]

            feature_counts = class_nodes.sum(dim=0).numpy()
            feature_distribution = feature_counts / feature_counts.sum()
            Fd.append(feature_distribution)

            num_features_per_node = class_nodes.sum(dim=1).numpy()
            feature_count_distribution = np.bincount(num_features_per_node.astype(int), minlength=self.feature_number)
            Md.append(feature_count_distribution / feature_count_distribution.sum())

        SA = [As]
        SX = [Xs]
        logits_query = self.net1(g, self.features)
        _, labels_query = torch.max(logits_query, dim=1)

        src, dst = As.nonzero(as_tuple=True)
        initial_num_nodes = Xs.shape[0]
        initial_graph = dgl.graph((src, dst), num_nodes=initial_num_nodes)
        initial_graph.ndata['feat'] = Xs

        self.net1.eval()
        initial_query = self.net1(initial_graph, initial_graph.ndata['feat'])
        _, initial_label = torch.max(initial_query, dim=1)

        SL = initial_label.tolist()
        n = 10

        for i in range(n):
            # For each class, generate and store a new sampled subgraph
            for c in range(self.label_number):
                num_nodes = As.shape[0]
                Ac = torch.ones((num_nodes, num_nodes))
                Xc = torch.zeros(num_nodes, len(Fd[c]))
                for i in range(num_nodes):
                    m = np.random.choice(np.arange(len(Md[c])), p=Md[c])
                    features = np.random.choice(len(Fd[c]), size=m, replace=False, p=Fd[c])
                    Xc[i, features] = 1
                SA.append(Ac)
                SX.append(Xc)

                src, dst = Ac.nonzero(as_tuple=True)
                subgraph = dgl.graph((src, dst), num_nodes=num_nodes)
                subgraph.ndata['feat'] = Xc

                self.net1.eval()
                api_query = self.net1(subgraph, subgraph.ndata['feat'])
                _, label_query = torch.max(api_query, dim=1)

                SL.extend(label_query.tolist())

        AG_list = [dense_to_sparse(torch.tensor(a))[0] for a in SA]
        XG = torch.vstack([torch.tensor(x) for x in SX])

        SL = torch.tensor(SL, dtype=torch.long)

        # Filter out invalid labels (negative labels) and trim the labels to match the feature matrix size

        valid_mask = SL >= 0
        SL = SL[valid_mask]
        SL = SL[:XG.shape[0]]

        # Combine the edge indices of all subgraphs, adjusting the node indices to avoid overlap
        AG_combined = torch.cat([edge_index + i * num_nodes for i, edge_index in enumerate(AG_list)],
                                dim=1)  # edge matrix

        src, dst = AG_combined[0], AG_combined[1]
        num_total_nodes = XG.shape[0]
        sub_g = dgl.graph((src, dst), num_nodes=num_total_nodes)
        sub_g.ndata['feat'] = XG

        net6 = GCN(XG.shape[1], self.label_number)

        optimizer = torch.optim.Adam(net6.parameters(), lr=0.01, weight_decay=5e-4)

        dur = []

        print("=========Model Extracting==========================")
        best_performance_metrics = GraphNeuralNetworkMetric()
        for epoch in tqdm(range(200)):
            if epoch >= 3:
                t0 = time.time()

            net6.train()
            logits = net6(sub_g, XG)
            out = torch.log_softmax(logits, dim=1)
            loss = F.nll_loss(out, SL)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if epoch >= 3:
                dur.append(time.time() - t0)
            focus_gnn_metrics = GraphNeuralNetworkMetric(
                0, 0, net6, g, self.features, self.test_mask, self.labels, labels_query)
            focus_gnn_metrics.evaluate()

            best_performance_metrics.fidelity = max(
                best_performance_metrics.fidelity, focus_gnn_metrics.fidelity)
            best_performance_metrics.accuracy = max(
                best_performance_metrics.accuracy, focus_gnn_metrics.accuracy)

        print("========================Final results:=========================================")
        print(best_performance_metrics)
