from abc import ABC, abstractmethod

import torch

from datasets import Dataset
from utils.hardware import get_device
from typing import Union, Optional


class BaseAttack(ABC):
    def __init__(self, dataset: Dataset, attack_node_fraction: float = None, model_path: str = None,
                 device: Optional[Union[str, torch.device]] = None):
        self.device = torch.device(device) if device else get_device()
        print(f"Using device: {self.device}")

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
        self.model_path = model_path

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
