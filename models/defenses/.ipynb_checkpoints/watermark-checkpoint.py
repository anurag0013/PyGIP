from ..base.defense import BaseDefense
from ...utils.metrics import GraphNeuralNetworkMetric
from ...utils.models import Gcn_Net

import networkx as nx
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch_geometric.utils import erdos_renyi_graph
from dgl.dataloading import NeighborSampler, NodeCollator
from dgl.nn import SAGEConv
import dgl
from tqdm import tqdm
import os
import inspect

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class graph_to_dataset:
    """
    A utility class to wrap a DGLGraph into a dataset-like structure.

    Main responsibilities:
    1. Extract basic graph information such as the number of nodes, features, labels, and attack nodes.
    2. Retrieve and process node features, labels, as well as train/test masks.
    3. Ensure that all tensors are on the specified device (device).
    """
    def __init__(self, graph, attack_node_fraction, name=None):
        """
        Initializes graph_to_dataset.

        Parameters
        ----------
        graph : dgl.DGLGraph
            The input graph.
        attack_node_fraction : float
            The fraction of nodes designated for attack, used to compute self.attack_node_number.
        name : str, optional
            The name of the dataset (default is None).
        """
        self.graph = graph
        # Add self-loops to the graph
        self.graph = dgl.add_self_loop(self.graph)
        self.dataset_name = name

        # Basic graph information
        self.node_number = self.graph.num_nodes()
        self.feature_number = len(self.graph.ndata['feat'][0])
        self.label_number = int(max(self.graph.ndata['label']) - min(self.graph.ndata['label'])) + 1
        self.attack_node_number = int(self.node_number * attack_node_fraction)

        # Node features
        if isinstance(self.graph.ndata['feat'], torch.Tensor):
            self.features = self.graph.ndata['feat']
        else:
            self.features = torch.FloatTensor(self.graph.ndata['feat']).to(device)

        # Node labels
        if isinstance(self.graph.ndata['label'], torch.Tensor):
            self.labels = self.graph.ndata['label']
        else:
            self.labels = torch.LongTensor(self.graph.ndata['label']).to(device)

        # Train mask
        if isinstance(self.graph.ndata['train_mask'], torch.Tensor):
            self.train_mask = self.graph.ndata['train_mask']
        else:
            self.train_mask = torch.BoolTensor(self.graph.ndata['train_mask']).to(device)

        # Test mask
        if isinstance(self.graph.ndata['test_mask'], torch.Tensor):
            self.test_mask = self.graph.ndata['test_mask']
        else:
            self.test_mask = torch.BoolTensor(self.graph.ndata['test_mask']).to(device)

        # Ensure all data is on the correct device
        if device != 'cpu':
            self.features = self.features.to(device)
            self.labels = self.labels.to(device)
            self.train_mask = self.train_mask.to(device)
            self.test_mask = self.test_mask.to(device)


class WatermarkGraph:
    """
    A class for generating a watermark graph.

    This class first uses the 'erdos_renyi_graph' function to randomly generate edges,
    then randomly generates node features and labels, and finally creates a DGLGraph
    with the generated information.
    """
    def __init__(self, n, num_features, num_classes, pr=0.1, pg=0, device=device):
        """
        Initializes WatermarkGraph.

        Parameters
        ----------
        n : int
            The number of nodes in the watermark graph.
        num_features : int
            The dimension of node features.
        num_classes : int
            The total number of classes in the dataset.
        pr : float, optional
            The probability of having 1 in node features (via binomial distribution). Default is 0.1.
        pg : float, optional
            Probability of connecting edges in Erdos-Renyi graph. Default is 0.
        device : torch.device, optional
            The device to run on, default is the global 'device'.
        """
        self.pr = pr
        self.pg = pg
        self.device = device
        self.graph_wm = self._generate_wm(n, num_features, num_classes)

    def _generate_wm(self, n, num_features, num_classes):
        """
        Internal function to generate the watermark graph.

        Parameters
        ----------
        n : int
            The number of nodes.
        num_features : int
            The dimension of node features.
        num_classes : int
            The total number of classes.

        Returns
        -------
        dgl.DGLGraph
            A watermark graph with random features and labels.
        """
        wm_edge_index = erdos_renyi_graph(n, self.pg, directed=False)
        wm_x = torch.tensor(np.random.binomial(
            1, self.pr, size=(n, num_features)), dtype=torch.float32).to(self.device)
        wm_y = torch.tensor(np.random.randint(
            low=0, high=num_classes, size=n), dtype=torch.long).to(self.device)

        data = dgl.graph((wm_edge_index[0], wm_edge_index[1]), num_nodes=n)
        data = data.to(self.device)

        data.ndata['feat'] = wm_x
        data.ndata['label'] = wm_y
        return data


class GraphSAGE(nn.Module):
    """
    A GraphSAGE model implemented with PyG's SAGEConv module.

    It consists of two SAGEConv layers:
    - The first layer projects features to 'hidden_channels',
    - The second layer outputs 'out_channels'.
    """
    def __init__(self, in_channels, hidden_channels, out_channels):
        """
        Initializes the GraphSAGE model.

        Parameters
        ----------
        in_channels : int
            The dimensionality of the input features.
        hidden_channels : int
            The dimensionality of the hidden layer.
        out_channels : int
            The dimensionality of the output layer (or the number of classes).
        """
        super(GraphSAGE, self).__init__()
        self.conv1 = SAGEConv(in_channels, hidden_channels, aggregator_type='mean')
        self.conv2 = SAGEConv(hidden_channels, out_channels, aggregator_type='mean')

    def forward(self, blocks, x):
        """
        Forward pass.

        Parameters
        ----------
        blocks : list of dgl.DGLGraph
            A list of subgraphs sampled for multiple layers.
        x : torch.Tensor
            The node features of shape (num_nodes, in_channels).

        Returns
        -------
        torch.Tensor
            The model outputs (logits) of shape (num_nodes, out_channels).
        """
        x = self.conv1(blocks[0], x)
        x = F.relu(x)
        x = self.conv2(blocks[1], x)
        return x
    
class Defense:
    """
    A base class for defense (or watermark) operations.
    In this example, it is mainly used to merge watermark graphs with
    existing datasets or perform subsequent model extraction attacks.
    """
    def __init__(self, dataset, attack_node_fraction):
        """
        Initializes the Defense class.

        Parameters
        ----------
        dataset : graph_to_dataset
            A dataset object that contains a graph, features, labels, and masks.
        attack_node_fraction : float
            The fraction of attack nodes in the total number of nodes, used later on.
        """
        self.dataset = dataset
        self.graph = dataset.graph

        self.node_number = dataset.node_number
        self.feature_number = dataset.feature_number
        self.label_number = dataset.label_number
        self.attack_node_number = int(dataset.node_number * attack_node_fraction)

        self.features = dataset.features
        self.labels = dataset.labels

        self.train_mask = dataset.train_mask
        self.test_mask = dataset.test_mask

    def train(self, loader):
        """
        A sample training function that iterates over the given DataLoader (blocks) to perform gradient updates.

        Parameters
        ----------
        loader : DataLoader
            A DGL batch training DataLoader that yields blocks.

        Returns
        -------
        float
            The average loss of the current training iteration.
        """
        self.model.train()
        total_loss = 0
        for _, _, blocks in loader:
            blocks = [b.to(self.device) for b in blocks]
            input_features = blocks[0].srcdata['feat']
            output_labels = blocks[-1].dstdata['label']

            self.optimizer.zero_grad()
            output_predictions = self.model(blocks, input_features)
            loss = F.cross_entropy(output_predictions, output_labels)
            loss.backward()
            self.optimizer.step()
            total_loss += loss.item()
        return total_loss / len(loader)

    def test(self, loader):
        """
        A sample test/validation function.

        Parameters
        ----------
        loader : DataLoader
            A DataLoader for test/validation, containing blocks.

        Returns
        -------
        float
            The accuracy on the test/validation set.
        """
        self.model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for _, _, blocks in loader:
                blocks = [b.to(self.device) for b in blocks]
                input_features = blocks[0].srcdata['feat']
                output_labels = blocks[-1].dstdata['label']
                output_predictions = self.model(blocks, input_features)
                pred = output_predictions.argmax(dim=1)
                correct += (pred == output_labels).sum().item()
                total += len(output_labels)
        return correct / total

    def merge_cora_and_datawm(self, cora_graph, datawm):
        """
        Merges the original graph (cora_graph) and the watermark graph (datawm).

        Parameters
        ----------
        cora_graph : dgl.DGLGraph
            The original graph (e.g., Cora dataset).
        datawm : dgl.DGLGraph
            The watermark graph.

        Returns
        -------
        dgl.DGLGraph
            The merged graph, containing newly added nodes and edges. 
            Also includes an updated watermark mask 'wm_mask'.
        """
        device = cora_graph.device
        datawm = datawm.to(device)

        num_cora_nodes = cora_graph.number_of_nodes()
        num_wm_nodes = datawm.number_of_nodes()

        cora_feat_dim = cora_graph.ndata['feat'].shape[1]
        wm_feat_dim = datawm.ndata['feat'].shape[1]

        # If feature dimensions differ, perform padding or truncation
        if cora_feat_dim != wm_feat_dim:
            if cora_feat_dim > wm_feat_dim:
                padding = torch.zeros(num_wm_nodes, cora_feat_dim - wm_feat_dim, device=device)
                datawm.ndata['feat'] = torch.cat([datawm.ndata['feat'], padding], dim=1)
            else:
                datawm.ndata['feat'] = datawm.ndata['feat'][:, :cora_feat_dim]

        # Align ndata keys of datawm with cora_graph
        for key in cora_graph.ndata.keys():
            if key not in datawm.ndata:
                if key in ['train_mask', 'val_mask', 'test_mask']:
                    datawm.ndata[key] = torch.zeros(num_wm_nodes, dtype=torch.bool, device=device)
                elif key == 'norm':
                    datawm.ndata[key] = torch.ones(num_wm_nodes, 1, dtype=torch.float32, device=device)
                else:
                    shape = (num_wm_nodes,) + cora_graph.ndata[key].shape[1:]
                    datawm.ndata[key] = torch.zeros(shape, dtype=cora_graph.ndata[key].dtype, device=device)

        merged_graph = dgl.batch([cora_graph, datawm])

        wm_mask = torch.zeros(num_cora_nodes + num_wm_nodes, dtype=torch.bool, device=device)
        wm_mask[num_cora_nodes:] = True
        merged_graph.ndata['wm_mask'] = wm_mask

        return merged_graph

    def generate_extended_label_file(self, datasetCora_merge, original_node_count, new_node_count, output_file):
        """
        Generates or extends a label file for subsequent attacks.

        Parameters
        ----------
        datasetCora_merge : graph_to_dataset
            The merged dataset object after watermark insertion.
        original_node_count : int
            The number of nodes in the original dataset.
        new_node_count : int
            The number of new watermark nodes.
        output_file : str
            The path to the label file to be written.
        """
        labels = datasetCora_merge.labels
        num_classes = len(torch.unique(datasetCora_merge.labels))

        file_exists = os.path.isfile(output_file)
        with open(output_file, 'w') as f:
            # Write labels for original nodes
            for i in range(original_node_count):
                f.write(f"{i} {labels[i]}\n")

            # Write labels for the new nodes
            import random
            for i in range(original_node_count, original_node_count + new_node_count):
                new_label = random.randint(0, num_classes - 1)
                f.write(f"{i} {new_label}\n")

    def watermark_attack(self, dataset, attack_name, dataset_name):
        """
        Injects watermark into the dataset and executes a specified attack method.

        Parameters
        ----------
        dataset : graph_to_dataset
            The original dataset wrapper.
        attack_name : int
            The ID of the attack method (corresponds to ModelExtractionAttackX classes).
        dataset_name : int
            The dataset ID (e.g., 1 for cora, 2 for citeseer, 3 for pubmed).
        """
        # 1) Insert/train watermark into the original dataset
        datasetCora = Watermark_sage(dataset, 0.25)
        datasetCora.attack()

        # 2) Merge the original graph with the watermark graph
        graph = datasetCora.merge_cora_and_datawm(datasetCora.graph, datasetCora.datawm)
        datasetCora_merge = graph_to_dataset(graph, 0.25, dataset.dataset_name)

        defense_path = inspect.getfile(Defense)
        flag = False

        # Select the attack implementation based on attack_name
        if (dataset_name == 1):
            if (attack_name == 1):
                attack = ModelExtractionAttack0(datasetCora_merge, 0.25)
                attack.attack()
                flag = True
            elif (attack_name == 2):
                # Example for the Cora dataset
                original_node_count = 2708
                new_node_count = 50
                output_file = os.path.abspath(os.path.join(defense_path, 
                    "../../../pygip/data/attack2_generated_graph/cora/query_labels_cora.txt"))
                print("Current working directory:", output_file)
                self.generate_extended_label_file(datasetCora_merge, original_node_count, new_node_count, output_file)

                attack = ModelExtractionAttack1(
                    datasetCora_merge, 
                    0.25, 
                    os.path.abspath(os.path.join(defense_path, "../../../pygip/data/attack2_generated_graph/cora/selected_index.txt")),
                    os.path.abspath(os.path.join(defense_path, "../../../pygip/data/attack2_generated_graph/cora/query_labels_cora.txt")),
                    os.path.abspath(os.path.join(defense_path, "../../../pygip/data/attack2_generated_graph/cora/graph_label0_564_541.txt"))
                )
                attack.attack()
                flag = True
            elif (attack_name == 3):
                attack = ModelExtractionAttack2(datasetCora_merge, 0.25)
                attack.attack()
                flag = True
            elif (attack_name == 4):
                attack = ModelExtractionAttack3(datasetCora_merge, 0.25)
                attack.attack()
                flag = True
            elif (attack_name == 5):
                attack = ModelExtractionAttack4(
                    datasetCora_merge, 0.25, 
                    os.path.abspath(os.path.join(defense_path, 
                        "../../../pygip/models/attack_3_subgraph_shadow_model_cora_8159.pkl"))
                )
                attack.attack()
                flag = True
            else:
                attack = ModelExtractionAttack5(
                    datasetCora_merge, 0.25, 
                    os.path.abspath(os.path.join(defense_path, 
                        "../../../pygip/models/attack_3_subgraph_shadow_model_cora_8159.pkl"))
                )
                attack.attack()
                flag = True

        elif (dataset_name == 2):
            # Citeseer
            if (attack_name == 1):
                attack = ModelExtractionAttack0(datasetCora_merge, 0.25)
                attack.attack()
                flag = True
            elif (attack_name == 2):
                original_node_count = 3327
                new_node_count = 50
                output_file = os.path.abspath(os.path.join(defense_path, 
                    "../../../pygip/data/attack2_generated_graph/citeseer/query_labels_citeseer.txt"))
                self.generate_extended_label_file(datasetCora_merge, original_node_count, new_node_count, output_file)

                attack = ModelExtractionAttack1(
                    datasetCora_merge, 
                    0.25, 
                    os.path.abspath(os.path.join(defense_path, 
                        "../../../pygip/data/attack2_generated_graph/citeseer/selected_index.txt")),
                    os.path.abspath(os.path.join(defense_path, 
                        "../../../pygip/data/attack2_generated_graph/citeseer/query_labels_citeseer.txt")),
                    os.path.abspath(os.path.join(defense_path, 
                        "../../../pygip/data/attack2_generated_graph/citeseer/graph_label0_604_525.txt"))
                )
                attack.attack()
                flag = True
            elif (attack_name == 3):
                attack = ModelExtractionAttack2(datasetCora_merge, 0.25)
                attack.attack()
                flag = True
            elif (attack_name == 4):
                attack = ModelExtractionAttack3(datasetCora_merge, 0.25)
                attack.attack()
                flag = True
            elif (attack_name == 5):
                attack = ModelExtractionAttack4(
                    datasetCora_merge, 0.25, 
                    os.path.abspath(os.path.join(defense_path, 
                        '../../../pygip/models/attack_3_subgraph_shadow_model_citeseer_6966.pkl'))
                )
                attack.attack()
                flag = True
            else:
                attack = ModelExtractionAttack5(
                    datasetCora_merge, 0.25, 
                    os.path.abspath(os.path.join(defense_path, 
                        '../../../pygip/models/attack_3_subgraph_shadow_model_citeseer_6966.pkl'))
                )
                attack.attack()
                flag = True

        elif (dataset_name == 3):
            # Pubmed
            if (attack_name == 1):
                attack = ModelExtractionAttack0(datasetCora_merge, 0.25)
                attack.attack()
                flag = True
            elif (attack_name == 2):
                original_node_count = 19717
                new_node_count = 50
                output_file = os.path.abspath(os.path.join(defense_path, 
                    "../../../pygip/data/attack2_generated_graph/pubmed/query_labels_pubmed.txt"))
                self.generate_extended_label_file(datasetCora_merge, original_node_count, new_node_count, output_file)

                attack = ModelExtractionAttack1(
                    datasetCora_merge, 
                    0.25, 
                    os.path.abspath(os.path.join(defense_path, 
                        "../../../pygip/data/attack2_generated_graph/pubmed/selected_index.txt")),
                    os.path.abspath(os.path.join(defense_path, 
                        "../../../pygip/data/attack2_generated_graph/pubmed/query_labels_pubmed.txt")),
                    os.path.abspath(os.path.join(defense_path, 
                        "../../../pygip/data/attack2_generated_graph/pubmed/graph_label0_657_667.txt"))
                )
                attack.attack()
                flag = True
            elif (attack_name == 3):
                attack = ModelExtractionAttack2(datasetCora_merge, 0.25)
                attack.attack()
                flag = True
            elif (attack_name == 4):
                attack = ModelExtractionAttack3(datasetCora_merge, 0.25)
                attack.attack()
                flag = True
            elif (attack_name == 5):
                attack = ModelExtractionAttack4(
                    datasetCora_merge, 0.25, 
                    '../../../pygip/models/attack_3_subgraph_shadow_model_pubmed_8063.pkl'
                )
                attack.attack()
                flag = True
            else:
                flag = True
                attack = ModelExtractionAttack5(
                    datasetCora_merge, 0.25, 
                    '../../../pygip/models/attack_3_subgraph_shadow_model_pubmed_8063.pkl'
                )
                attack.attack()

        # If the attack was successfully executed, evaluate watermark accuracy
        if (flag == True):
            datawm = datasetCora.datawm
            datasetCora_wm = graph_to_dataset(datawm, 0.25, dataset.dataset_name)
            datasetCora_wm.test_mask = torch.ones_like(datasetCora_wm.test_mask, dtype=torch.bool)

            net = Gcn_Net(attack.feature_number, attack.label_number)
            evaluation = GraphNeuralNetworkMetric(
                0, 0, net, datasetCora_wm.graph, datasetCora_wm.features, datasetCora_wm.test_mask, datasetCora_wm.labels
            )
            evaluation.evaluate()
            print("Watermark Graph - Accuracy:", evaluation.accuracy)
            
class Watermark_sage(Defense):
    """
    Inherits from Defense. Uses GraphSAGE to train on both the original and 
    watermark graphs, ultimately embedding a watermark into the trained model (or graph).
    """
    def __init__(self, dataset, attack_node_fraction, wm_node=50, pr=0.1, pg=0, device=device):
        """
        Initializes Watermark_sage.

        Parameters
        ----------
        dataset : graph_to_dataset
            The original dataset.
        attack_node_fraction : float
            The fraction of attack nodes.
        wm_node : int, optional
            The number of nodes in the watermark graph (default is 50).
        pr : float, optional
            Probability of having 1 in the features of watermark nodes. Default is 0.1.
        pg : float, optional
            Edge probability for Erdos-Renyi watermark graph. Default is 0.
        device : torch.device, optional
            Device to run on.
        """
        super().__init__(dataset, attack_node_fraction)
        self.wm_node = wm_node
        self.pr = pr
        self.pg = pg
        self.device = device

        # GraphSAGE-based model
        self.model = GraphSAGE(in_channels=self.graph.ndata['feat'].shape[1],
                               hidden_channels=128,
                               out_channels=7)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=0.01, weight_decay=5e-4)

    def erdos_renyi_graph(self, n, p, directed=False):
        """
        A custom Erdos-Renyi random graph generator.

        Parameters
        ----------
        n : int
            Number of nodes.
        p : float
            Probability of connecting two nodes.
        directed : bool, optional
            Whether the graph is directed (default is False).

        Returns
        -------
        torch.Tensor
            An edge index tensor of shape (2, num_edges).
        """
        edges = []
        for i in range(n):
            for j in range(i+1, n):
                if torch.rand(1).item() < p:
                    edges.append([i, j])
                    if not directed:
                        edges.append([j, i])

        # Ensure at least one edge exists
        if not edges:
            i, j = torch.randint(0, n, (2,))
            while i == j:
                j = torch.randint(0, n, (1,))
            edges = [[i.item(), j.item()], [j.item(), i.item()]]

        edges = torch.tensor(edges, dtype=torch.long).t()
        return edges

    def attack(self):
        """
        Main attack (or watermark injection) process:
        1. Generate the watermark graph (data_wm).
        2. Pre-train on the original graph via GraphSAGE.
        3. Optionally fine-tune on the watermark graph.
        """
        # Generate watermark graph
        data_wm = WatermarkGraph(
            n=self.wm_node,
            num_features=self.feature_number,
            num_classes=self.label_number,
            pr=self.pr,
            pg=self.pg,
            device=self.device
        ).graph_wm

        self.datawm = data_wm

        # Move all data to the designated device
        self.graph = self.graph.to(self.device)
        self.features = self.features.to(self.device)
        self.labels = self.labels.to(self.device)
        self.model = self.model.to(self.device)

        data_wm = data_wm.to(self.device)
        data_wm.ndata['feat'] = data_wm.ndata['feat'].to(self.device)
        data_wm.ndata['label'] = data_wm.ndata['label'].to(self.device)

        # Construct samplers and DataLoaders
        sampler = NeighborSampler([5, 5])
        train_nids = self.graph.ndata['train_mask'].nonzero(as_tuple=True)[0].to(self.device)
        # If 'val_mask' doesn't exist, use 'test_mask' instead (for demonstration)
        val_nids = self.graph.ndata.get('val_mask', self.graph.ndata['test_mask']).nonzero(as_tuple=True)[0].to(self.device)
        test_nids = self.graph.ndata['test_mask'].nonzero(as_tuple=True)[0].to(self.device)
        wm_nids = torch.arange(data_wm.number_of_nodes(), device=self.device)

        train_collator = NodeCollator(self.graph.to(self.device), train_nids, sampler)
        val_collator = NodeCollator(self.graph.to(self.device), val_nids, sampler)
        test_collator = NodeCollator(self.graph.to(self.device), test_nids, sampler)
        wm_collator = NodeCollator(data_wm.to(self.device), wm_nids, sampler)

        train_dataloader = DataLoader(
            train_collator.dataset,
            batch_size=32,
            shuffle=True,
            collate_fn=train_collator.collate,
            drop_last=False
        )

        val_dataloader = DataLoader(
            val_collator.dataset,
            batch_size=32,
            shuffle=False,
            collate_fn=val_collator.collate,
            drop_last=False
        )

        test_dataloader = DataLoader(
            test_collator.dataset,
            batch_size=32,
            shuffle=False,
            collate_fn=test_collator.collate,
            drop_last=False
        )

        wm_dataloader = DataLoader(
            wm_collator.dataset,
            batch_size=self.wm_node,
            shuffle=False,
            collate_fn=wm_collator.collate,
            drop_last=False
        )

        # Ensure graph features and labels are on the correct device
        self.graph.ndata['feat'] = self.graph.ndata['feat'].to(self.device)
        self.graph.ndata['label'] = self.graph.ndata['label'].to(self.device)

        # 1st stage: Train on the original graph
        for epoch in tqdm(range(1, 51)):
            loss = self.train(train_dataloader)
            val_acc = self.test(val_dataloader)
            test_acc = self.test(test_dataloader)
            # Optionally, record or print loss, val_acc, test_acc here

        # Evaluate on watermark data before fine-tuning
        nonmarked_acc = self.test(wm_dataloader)

        # Evaluate on the original test set
        marked_acc = self.test(test_dataloader)
        print(f'Marked Acc: {marked_acc:.4f}')

        # 2nd stage: Fine-tune on the watermark dataset
        for epoch in tqdm(range(1, 16)):
            loss = self.train(wm_dataloader)
            test_acc = self.test(wm_dataloader)
            # Optionally, record or print loss, test_acc here

        # Final results
        marked_acc = self.test(test_dataloader)
        watermark_acc = self.test(wm_dataloader)
        print('Final results')
        print('Non-Marked Acc: {:.4f}, Marked Acc: {:.4f}, Watermark Acc: {:.4f}'.format(
            nonmarked_acc, marked_acc, watermark_acc))
