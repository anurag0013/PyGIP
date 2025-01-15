# pygip/models/attacks/surrogate_extraction.py

from ..base.attack import Attack
import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from torch_geometric.data import Data

class SurrogateExtractionAttack(Attack):
    """
    Implementation of Surrogate Extraction Attack for GNN models
    """
    def __init__(self, hidden_dim=32, dropout_rate=0.5, alpha=0.5, epochs=300):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.dropout_rate = dropout_rate
        self.alpha = alpha
        self.epochs = epochs
        
    def attack(self, target_model, data, attack_node_counts):
        """
        Execute surrogate extraction attack
        
        Args:
            target_model: Target GNN model
            data: PyG Data object
            attack_node_counts: List of numbers of attack nodes to try
            
        Returns:
            dict: Fidelity results for each attack node count
        """
        self.data = data
        self.input_dim = data.x.shape[1]
        self.output_dim = len(torch.unique(data.y))
        
        degrees = torch.bincount(data.edge_index[0], minlength=data.num_nodes)
        
        results = {}
        for count in attack_node_counts:
            # Select attack nodes based on degree
            threshold = torch.quantile(degrees.float(), 0.5)
            low_degree_nodes = (degrees <= threshold).nonzero(as_tuple=True)[0]
            
            if len(low_degree_nodes) >= count:
                attack_nodes = low_degree_nodes[torch.randperm(len(low_degree_nodes))[:count]]
            else:
                self.logger.warning(f"Not enough low-degree nodes to select {count} attack nodes")
                attack_nodes = low_degree_nodes
                
            surrogate_data = self._construct_surrogate_graph(attack_nodes)
            surrogate_model = self._train_surrogate_model(surrogate_data)
            fidelity = self._evaluate_fidelity(target_model, surrogate_model)
            results[count] = fidelity * 100
            
        return results
    
    def _construct_surrogate_graph(self, attack_nodes):
        """Construct surrogate graph with synthetic nodes"""
        x = self.data.x.clone()
        edge_index = self.data.edge_index.clone()
        train_edges = self.data.train_mask[edge_index[0]] & self.data.train_mask[edge_index[1]]
        edge_index = edge_index[:, train_edges]
        degrees = torch.bincount(edge_index[0], minlength=self.data.num_nodes)

        synthetic_nodes = []
        synthetic_edges = []

        for node in attack_nodes:
            # Get 1-hop and 2-hop neighbors
            neighbors_1hop = edge_index[1][edge_index[0] == node]
            neighbors_1hop = neighbors_1hop[self.data.train_mask[neighbors_1hop]]

            neighbors_2hop = set()
            for neighbor in neighbors_1hop:
                neighbors_of_neighbor = edge_index[1][edge_index[0] == neighbor]
                neighbors_2hop.update(
                    neighbors_of_neighbor[self.data.train_mask[neighbors_of_neighbor]].tolist()
                )
            neighbors_2hop -= set(neighbors_1hop.tolist())

            synthetic_node_id = len(x)
            synthetic_nodes.append(synthetic_node_id)
            synthetic_edges.extend([[node, synthetic_node_id], [synthetic_node_id, node]])

            # Compute synthetic node features
            feature_1hop = (
                sum(self.data.x[neighbor] / degrees[neighbor] for neighbor in neighbors_1hop) 
                / len(neighbors_1hop) if len(neighbors_1hop) > 0 
                else torch.zeros_like(self.data.x[0])
            )
            
            feature_2hop = (
                sum(self.data.x[neighbor] / degrees[neighbor] for neighbor in neighbors_2hop)
                / len(neighbors_2hop) if len(neighbors_2hop) > 0
                else torch.zeros_like(self.data.x[0])
            )
            
            synthetic_feature = feature_1hop + self.alpha * feature_2hop
            x = torch.cat([x, synthetic_feature.unsqueeze(0)], dim=0)

        if synthetic_edges:
            synthetic_edges = torch.tensor(synthetic_edges, dtype=torch.long).t().contiguous()
            edge_index = torch.cat([edge_index, synthetic_edges], dim=1)

        new_train_mask = torch.cat([
            self.data.train_mask, 
            torch.zeros(len(synthetic_nodes), dtype=torch.bool)
        ])
        new_y = torch.cat([
            self.data.y, 
            torch.full((len(synthetic_nodes),), -1)
        ])

        return Data(x=x, edge_index=edge_index, y=new_y, train_mask=new_train_mask)
    
    def _train_surrogate_model(self, data):
        """Train surrogate model"""
        from ...utils.models import Gcn_Net  # Use existing GCN implementation
        
        model = Gcn_Net(
            nfeat=self.input_dim,
            nhid=self.hidden_dim,
            nclass=self.output_dim,
            dropout=self.dropout_rate
        )
        
        optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)
        
        model.train()
        for _ in range(self.epochs):
            optimizer.zero_grad()
            out = model(data)
            valid_train_mask = data.train_mask & (data.y != -1)
            loss = F.nll_loss(out[valid_train_mask], data.y[valid_train_mask])
            loss.backward()
            optimizer.step()
            
        return model
    
    def _evaluate_fidelity(self, target_model, surrogate_model):
        """Evaluate surrogate model fidelity"""
        target_model.eval()
        surrogate_model.eval()
        with torch.no_grad():
            target_preds = target_model(self.data).argmax(dim=1)
            surrogate_preds = surrogate_model(self.data).argmax(dim=1)
            return (target_preds == surrogate_preds).sum().item() / len(target_preds)