from abc import abstractmethod

import dgl
import networkx as nx
import torch
import torch.nn.functional as F
from tqdm import tqdm

from models.attack.base import BaseAttack
from models.nn import GCN, GraphSAGE  # Backbone architectures


class GraphGenerator:
    def __init__(self, node_number, feature_number, label_number):
        self.node_number = node_number
        self.feature_number = feature_number
        self.label_number = label_number

    def generate(self):
        # Generate a random Erdős–Rényi graph and convert to DGL
        g_nx = nx.erdos_renyi_graph(n=self.node_number, p=0.05)
        g_dgl = dgl.from_networkx(g_nx)
        # Random node features
        features = torch.randn((self.node_number, self.feature_number))
        return g_dgl, features


class DFEAAttack(BaseAttack):
    """
    Data-Free Extraction Attack base class.
    Handles victim model training/loading and synthetic graph generation.
    """

    def __init__(self, dataset, attack_node_fraction, model_path=None):
        assert dataset.api_type == 'dgl', "only support dgl api"
        super().__init__(dataset, attack_node_fraction, model_path)
        # load graph data
        self.graph = dataset.graph_data
        self.features = self.graph.ndata['feat']
        self.labels = self.graph.ndata['label']
        self.train_mask = self.graph.ndata['train_mask']
        self.val_mask = self.graph.ndata['val_mask']
        # meta data
        self.feature_number = dataset.num_features
        self.label_number = dataset.num_classes
        self.attack_node_number = int(dataset.num_nodes * attack_node_fraction)
        # Generate synthetic graph and features for surrogate training
        self.generator = GraphGenerator(
            node_number=self.attack_node_number,
            feature_number=self.feature_number,
            label_number=self.label_number
        )
        self.synthetic_graph, self.synthetic_features = self.generator.generate()
        if model_path is None:
            self._train_target_model()
        else:
            self._load_model(model_path)

    def _train_target_model(self):
        # Train the victim GCN model on real data (mirroring main.py)
        model = GCN(self.feature_number, self.label_number)
        optimizer = torch.optim.Adam(
            model.parameters(), lr=0.01, weight_decay=5e-4
        )
        model.train()
        # Identify dataset for label shaping
        name = getattr(self.dataset, 'dataset_name', None) or getattr(self.dataset, 'name', None)
        epochs = 200
        for epoch in range(1, epochs + 1):
            optimizer.zero_grad()
            logits = model(self.graph, self.features)
            labels = self.labels.squeeze() if name == 'ogb-arxiv' else self.labels
            loss = F.nll_loss(
                F.log_softmax(logits[self.train_mask], dim=1),
                labels[self.train_mask]
            )
            loss.backward()
            optimizer.step()

        model.eval()
        self.model = model

    def _load_model(self, model_path):
        # Load a pretrained victim model
        model = GCN(self.feature_number, self.label_number)
        state = torch.load(model_path, map_location=torch.device('cpu'))
        model.load_state_dict(state)
        model.eval()
        self.model = model

    def _forward(self, model, graph, features):
        # Abstract forward for GCN and GraphSAGE
        if isinstance(model, GraphSAGE):
            # GraphSAGE expects two-block input list
            return model([graph, graph], features)
        return model(graph, features)

    def evaluate(self, surrogate):
        # Compute agreement accuracy between surrogate and victim on synthetic data
        surrogate.eval()
        with torch.no_grad():
            logits_s = self._forward(
                surrogate, self.synthetic_graph, self.synthetic_features
            )
            preds_s = logits_s.argmax(dim=1)
            logits_v = self._forward(
                self.model, self.synthetic_graph, self.synthetic_features
            )
            preds_v = logits_v.argmax(dim=1)
        return (preds_s == preds_v).float().mean().item()

    @abstractmethod
    def attack(self):
        pass


class DFEATypeI(DFEAAttack):
    """
    Type I: Uses victim outputs + gradients for surrogate training.
    """

    def attack(self):
        surrogate = GCN(self.feature_number, self.label_number)
        optimizer = torch.optim.Adam(surrogate.parameters(), lr=0.01)
        for _ in tqdm(range(200)):
            surrogate.train()
            optimizer.zero_grad()
            # Victim logits (no gradient)
            with torch.no_grad():
                logits_v = self._forward(
                    self.model, self.synthetic_graph, self.synthetic_features
                )
            logits_s = self._forward(
                surrogate, self.synthetic_graph, self.synthetic_features
            )
            loss = F.kl_div(
                F.log_softmax(logits_s, dim=1),
                F.softmax(logits_v, dim=1),
                reduction='batchmean'
            )
            loss.backward()
            optimizer.step()
        metric = self.evaluate(surrogate)
        print('Agreement Acc: ', metric)
        return metric


class DFEATypeII(DFEAAttack):
    """
    Type II: Uses victim outputs only (hard labels).
    """

    def attack(self):
        surrogate = GraphSAGE(self.feature_number, 16, self.label_number)
        optimizer = torch.optim.Adam(surrogate.parameters(), lr=0.01)
        for _ in tqdm(range(200)):
            surrogate.train()
            optimizer.zero_grad()
            with torch.no_grad():
                logits_v = self._forward(
                    self.model, self.synthetic_graph, self.synthetic_features
                )
            logits_s = self._forward(
                surrogate, self.synthetic_graph, self.synthetic_features
            )
            pseudo = logits_v.argmax(dim=1)
            loss = F.cross_entropy(logits_s, pseudo)
            loss.backward()
            optimizer.step()
        metric = self.evaluate(surrogate)
        print('Agreement Acc: ', metric)
        return metric


class DFEATypeIII(DFEAAttack):
    """
    Type III: Two surrogates with victim supervision + consistency.
    """

    def attack(self):
        s1 = GCN(self.feature_number, self.label_number)
        s2 = GraphSAGE(self.feature_number, 16, self.label_number)
        opt1 = torch.optim.Adam(s1.parameters(), lr=0.01)
        opt2 = torch.optim.Adam(s2.parameters(), lr=0.01)
        for _ in tqdm(range(200)):
            s1.train()
            s2.train()
            opt1.zero_grad()
            opt2.zero_grad()
            # Victim pseudo-labels
            with torch.no_grad():
                logits_v = self._forward(
                    self.model, self.synthetic_graph, self.synthetic_features
                )
            pseudo_v = logits_v.argmax(dim=1)
            # Surrogate predictions
            l1 = self._forward(s1, self.synthetic_graph, self.synthetic_features)
            l2 = self._forward(s2, self.synthetic_graph, self.synthetic_features)
            # Loss: supervised + consistency
            loss1 = F.cross_entropy(l1, pseudo_v)
            loss2 = F.cross_entropy(l2, pseudo_v)
            cons = F.mse_loss(l1, l2)
            total = loss1 + loss2 + 0.5 * cons
            total.backward()
            opt1.step()
            opt2.step()
        metric = self.evaluate(s1)
        print('Agreement Acc: ', metric)
        return metric


# Factory mapping of attack names to classes
ATTACK_FACTORY = {
    "ModelExtractionAttack0": DFEATypeI,
    "ModelExtractionAttack1": DFEATypeI,
    "ModelExtractionAttack2": DFEATypeII,
    "ModelExtractionAttack3": DFEATypeIII
}
