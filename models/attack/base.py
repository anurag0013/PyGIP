from abc import ABC, abstractmethod

from datasets import Dataset


class BaseAttack(ABC):
    def __init__(self, dataset: Dataset, attack_node_fraction: float, model_path: str = None):
        """Base class for all attack implementations."""
        self.dataset = dataset
        self.graph = dataset.graph
        self.node_number = dataset.node_number
        self.feature_number = dataset.feature_number
        self.label_number = dataset.label_number
        self.attack_node_number = int(dataset.node_number * attack_node_fraction)
        self.attack_node_fraction = attack_node_fraction

        self.features = dataset.features
        self.labels = dataset.labels
        self.train_mask = dataset.train_mask
        self.test_mask = dataset.test_mask

        if model_path is None:
            self._train_target_model()
        else:
            self._load_model(model_path)

    @abstractmethod
    def attack(self):
        """
        Execute the attack.
        """
        raise NotImplementedError

    def _load_model(self, model_path):
        """
        Load a pre-trained model.
        """
        raise NotImplementedError

    def _train_target_model(self):
        """
        Train the target model if not provided.
        """
        raise NotImplementedError

    def _train_attack_model(self):
        """
        Train the attack model.
        """
        raise NotImplementedError
