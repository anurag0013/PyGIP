from abc import ABC, abstractmethod

from datasets import Dataset


class BaseDefense(ABC):
    def __init__(self, dataset: Dataset, attack_node_fraction: float):
        """Base class for all defense implementations."""
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

    @abstractmethod
    def defend(self):
        """Execute the defense mechanism."""
        raise NotImplementedError

    def evaluate(self):
        """Evaluate the effectiveness of the defense."""
        pass
