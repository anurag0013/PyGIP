
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
    
    def __init__(self, noise_dim, num_nodes, feature_dim, 
                 generator_lr=1e-6, surrogate_lr=0.001,
                 n_generator_steps=2, n_surrogate_steps=5):
        super().__init__()
        self.noise_dim = noise_dim
        self.num_nodes = num_nodes
        self.feature_dim = feature_dim
        self.generator_lr = generator_lr
        self.surrogate_lr = surrogate_lr
        self.n_generator_steps = n_generator_steps
        self.n_surrogate_steps = n_surrogate_steps
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        self.criterion = nn.CrossEntropyLoss()
        self.generator = None
        self.surrogate_model = None
        self.victim_model = None

    def initialize_models(self):
        """Initialize generator and surrogate models."""
        raise NotImplementedError
        
    def train(self, victim_model, **kwargs):
        """Train the attack model using the victim model."""
        raise NotImplementedError
        
    def extract(self, query_data):
        """Extract predictions using trained surrogate model."""
        if self.surrogate_model is None:
            raise ValueError("Model must be trained before extraction")
        return self.surrogate_model(query_data)
        
    def evaluate(self, test_data):
        """Evaluate attack performance."""
        self.surrogate_model.eval()
        with torch.no_grad():
            pred = self.surrogate_model(test_data.x, test_data.edge_index)
            acc = (pred.argmax(dim=1) == test_data.y).float().mean()
        return {'accuracy': acc.item()}
