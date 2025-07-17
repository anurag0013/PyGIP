from abc import ABC, abstractmethod

import torch
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score


class MetricBase(ABC):
    def __init__(self):
        self.preds = []
        self.labels = []
        self.reset()

    @abstractmethod
    def update(self, preds, labels, **kwargs):
        """Update internal metric state."""
        pass

    @abstractmethod
    def compute(self):
        """Compute and return all metric results."""
        pass

    @abstractmethod
    def reset(self):
        """Reset internal state."""
        self.preds = []
        self.labels = []

    @staticmethod
    def compute_default_metrics(preds, labels):
        return {
            'Acc': accuracy_score(labels, preds),
            'F1': f1_score(labels, preds, average='macro'),
            'Precision': precision_score(labels, preds, average='macro'),
            'Recall': recall_score(labels, preds, average='macro'),
            'AUROC': roc_auc_score(labels, preds, multi_class='ovr')
        }

    @staticmethod
    def compute_fidelity(output1, output2):
        return {
            'Fidelity': (output1.argmax(dim=1) == output2.argmax(dim=1)).float().mean().item()
        }

    @staticmethod
    def compute_auc(preds, labels):
        return {
            'AUC': roc_auc_score(labels, preds)
        }

    def print(self):
        results = self.compute()
        for name, value in results.items():
            print(f"{name}: {value:.4f}")


class AttackMetric(MetricBase):
    def __init__(self):
        super().__init__()

    def update(self, preds, labels, target_outputs=None, surrogate_outputs=None):
        self.preds.append(preds.detach().cpu())
        self.labels.append(labels.detach().cpu())

    def compute(self):
        preds = torch.cat(self.preds).numpy()
        labels = torch.cat(self.labels).numpy()
        results = self.compute_default_metrics(preds, labels)
        return results


class DefenseMetric(MetricBase):
    def __init__(self):
        super().__init__()
        self.target_preds = []
        self.target_labels = []
        self.surrogate_outputs = []

    def update_target(self, target_preds, target_labels):
        ...

    def update_defense_task(self, defense_preds, defense_labels):
        ...

    def update_defense_wm(self, wm_preds, wm_labels):
        ...

    def update_surrogate_task(self, surrogate_preds, surrogate_labels):
        ...

    def update_surrogate_wm(self, wm_preds, wm_labels):
        ...

    def update(self, preds, labels, target_outputs=None, surrogate_outputs=None):
        self.preds.append(preds.detach().cpu())
        self.labels.append(labels.detach().cpu())

    def compute(self):
        preds = torch.cat(self.preds).numpy()
        labels = torch.cat(self.labels).numpy()
        results = self.compute_default_metrics(preds, labels)
        return results


class GraphNeuralNetworkMetric:
    """
    Graph Neural Network Metric Class.

    This class evaluates two metrics, fidelity and accuracy, for a given
    GNN model on a specified graph and features.
    """

    def __init__(self, fidelity=0, accuracy=0, model=None,
                 graph=None, features=None, mask=None,
                 labels=None, query_labels=None):
        self.model = model if model is not None else None
        self.graph = graph if graph is not None else None
        self.features = features if features is not None else None
        self.mask = mask if mask is not None else None
        self.labels = labels if labels is not None else None
        self.query_labels = query_labels if query_labels is not None else None
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

    @staticmethod
    def calculate_surrogate_fidelity(target_model, surrogate_model, data, mask=None):
        """
        Calculate fidelity between target and surrogate model predictions.
        
        Args:
            target_model: Original model
            surrogate_model: Extracted surrogate model
            data: Input graph data
            mask: Optional mask for evaluation on specific nodes
            
        Returns:
            float: Fidelity score (percentage of matching predictions)
        """
        target_model.eval()
        surrogate_model.eval()

        with torch.no_grad():
            # Get predictions from both models
            target_logits = target_model(data)
            surrogate_logits = surrogate_model(data)

            # Apply mask if provided
            if mask is not None:
                target_logits = target_logits[mask]
                surrogate_logits = surrogate_logits[mask]

            # Get predicted classes
            target_preds = target_logits.argmax(dim=1)
            surrogate_preds = surrogate_logits.argmax(dim=1)

            # Calculate fidelity
            matches = (target_preds == surrogate_preds).sum().item()
            total = len(target_preds)

            return (matches / total) * 100

    @staticmethod
    def evaluate_surrogate_extraction(target_model, surrogate_model, data,
                                      train_mask=None, val_mask=None, test_mask=None):
        """
        Comprehensive evaluation of surrogate extraction attack.
        
        Args:
            target_model: Original model
            surrogate_model: Extracted surrogate model
            data: Input graph data
            train_mask: Mask for training nodes
            val_mask: Mask for validation nodes
            test_mask: Mask for test nodes
            
        Returns:
            dict: Dictionary containing fidelity scores for different data splits
        """
        results = {}

        # Overall fidelity
        results['overall_fidelity'] = GraphNeuralNetworkMetric.calculate_surrogate_fidelity(
            target_model, surrogate_model, data
        )

        # Split-specific fidelity if masks are provided
        if train_mask is not None:
            results['train_fidelity'] = GraphNeuralNetworkMetric.calculate_surrogate_fidelity(
                target_model, surrogate_model, data, train_mask
            )

        if val_mask is not None:
            results['val_fidelity'] = GraphNeuralNetworkMetric.calculate_surrogate_fidelity(
                target_model, surrogate_model, data, val_mask
            )

        if test_mask is not None:
            results['test_fidelity'] = GraphNeuralNetworkMetric.calculate_surrogate_fidelity(
                target_model, surrogate_model, data, test_mask
            )

        return results

    def __str__(self):
        """Returns a string representation of the metrics."""
        return f"Fidelity: {self.fidelity}, Accuracy: {self.accuracy}"
