from abc import ABC, abstractmethod

from datasets import Dataset


class BaseDefense(ABC):
    def __init__(self, dataset: Dataset, attack_node_fraction: float):
        # graph data
        self.dataset = dataset
        self.graph_dataset = dataset.graph_dataset
        self.graph_data = dataset.graph_data

        # meta data
        self.num_nodes = dataset.num_nodes
        self.num_features = dataset.num_features
        self.num_classes = dataset.num_classes

        # params
        self.attack_node_fraction = attack_node_fraction

    @abstractmethod
    def defend(self):
        """
        Execute the defense mechanism.
        """
        raise NotImplementedError

    def _load_model(self):
        """
        Load pre-trained model.
        """
        raise NotImplementedError

    def _train_target_model(self):
        """
        This is an optional method.
        """
        raise NotImplementedError

    def _train_defense_model(self):
        """
        This is an optional method.
        """
        raise NotImplementedError

    def _train_surrogate_model(self):
        """
        This is an optional method.
        """
        raise NotImplementedError
