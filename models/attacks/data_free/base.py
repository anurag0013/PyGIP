

from ...base.attack import BaseAttack
import torch
import torch.nn as nn

class DataFreeBaseAttack(BaseAttack):
    """Base class for data-free attacks.
    
    Attributes:
        generator: Graph generator model
        surrogate_model: Surrogate model for attack
        victim_model: Target victim model
        device: Computing device (CPU/GPU)
    """
    
    def __init__(self, generator_config, surrogate_config):
        """Initialize data-free attack base.
        
        Args:
            generator_config (dict): Configuration for generator model
            surrogate_config (dict): Configuration for surrogate model
        """
        super().__init__()
        self.generator_config = generator_config
        self.surrogate_config = surrogate_config
        self.generator = None
        self.surrogate_model = None
        self.victim_model = None
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'

    def initialize_models(self):
        """Initialize generator and surrogate models."""
        raise NotImplementedError
        
    def train(self, victim_model, **kwargs):
        """Train the attack models."""
        raise NotImplementedError
        
    def extract(self, query_data):
        """Extract knowledge using trained surrogate model."""
        raise NotImplementedError
        
    def evaluate(self, test_data):
        """Evaluate attack performance."""
        raise NotImplementedError
