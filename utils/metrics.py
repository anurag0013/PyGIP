import torch
import torch.nn.functional as F

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class GraphNeuralNetworkMetric:
    """
    Graph Neural Network Metric Class.

    This class evaluates two metrics, fidelity and accuracy, for a given
    GNN model on a specified graph and features.
    """
    def __init__(self, fidelity=0, accuracy=0, model=None,
                 graph=None, features=None, mask=None,
                 labels=None, query_labels=None):
        self.model = model.to(device) if model is not None else None
        self.graph = graph.to(device) if graph is not None else None
        self.features = features.to(device) if features is not None else None
        self.mask = mask.to(device) if mask is not None else None
        self.labels = labels.to(device) if labels is not None else None
        self.query_labels = query_labels.to(device) if query_labels is not None else None
        self.accuracy = accuracy
        self.fidelity = fidelity

    def evaluate_helper(self, model, graph, features, labels, mask):
        """Helper function to evaluate the model's performance."""
        if model is None or graph is None or features is None or labels is None or mask is None:
            return None
        model.eval()
        with torch.no_grad():
            logits = model(graph, features)
            logits = logits[mask]
            labels = labels[mask]
            _, indices = torch.max(logits, dim=1)
            correct = torch.sum(indices == labels)
        return correct.item() * 1.0 / len(labels)

    def evaluate(self):
        """Main function to update fidelity and accuracy scores."""
        self.accuracy = self.evaluate_helper(
            self.model, self.graph, self.features, self.labels, self.mask)
        self.fidelity = self.evaluate_helper(
            self.model, self.graph, self.features, self.query_labels, self.mask)

    def __str__(self):
        """Returns a string representation of the metrics."""
        return f"Fidelity: {self.fidelity}, Accuracy: {self.accuracy}"