from .data_free.base import DataFreeBaseAttack
from .data_free.type_i import TypeIAttack 
from .data_free.type_ii import TypeIIAttack
from .data_free.type_iii import TypeIIIAttack

class DataFreeAttack(DataFreeBaseAttack):
    """Data-free model extraction attack for GNNs.
    
    This class provides a unified interface for different types of
    data-free attacks on graph neural networks.
    
    Attributes:
        attack_type: Type of attack (1, 2, or 3)
        attack_impl: Specific attack implementation
    """
    
    def __init__(self, attack_type=1, noise_dim=32, num_nodes=500, feature_dim=64,
                 generator_lr=1e-6, surrogate_lr=0.001,
                 n_generator_steps=2, n_surrogate_steps=5):
        """Initialize data-free attack.
        
        Args:
            attack_type (int): Type of attack (1, 2, or 3)
            noise_dim (int): Dimension of input noise vector
            num_nodes (int): Number of nodes in generated graph
            feature_dim (int): Dimension of node features
            generator_lr (float): Learning rate for generator
            surrogate_lr (float): Learning rate for surrogate model
            n_generator_steps (int): Number of generator training steps
            n_surrogate_steps (int): Number of surrogate training steps
        """
        super().__init__(noise_dim, num_nodes, feature_dim,
                        generator_lr, surrogate_lr,
                        n_generator_steps, n_surrogate_steps)
        self.attack_type = attack_type
        self.attack_impl = None
        
    def initialize_models(self):
        """Initialize attack implementation based on type."""
        if self.attack_type == 1:
            self.attack_impl = TypeIAttack(
                self.noise_dim,
                self.num_nodes,
                self.feature_dim,
                self.generator_lr,
                self.surrogate_lr,
                self.n_generator_steps,
                self.n_surrogate_steps
            )
        elif self.attack_type == 2:
            self.attack_impl = TypeIIAttack(
                self.noise_dim,
                self.num_nodes,
                self.feature_dim,
                self.generator_lr,
                self.surrogate_lr,
                self.n_generator_steps,
                self.n_surrogate_steps
            )
        elif self.attack_type == 3:
            self.attack_impl = TypeIIIAttack(
                self.noise_dim,
                self.num_nodes,
                self.feature_dim,
                self.generator_lr,
                self.surrogate_lr,
                self.n_generator_steps,
                self.n_surrogate_steps
            )
        else:
            raise ValueError(f"Invalid attack type: {self.attack_type}")

    def train(self, victim_model, num_queries=300, log_interval=10):
        """Train the attack model.
        
        Args:
            victim_model: The target model to attack
            num_queries (int): Number of queries to make
            log_interval (int): Interval for logging progress
            
        Returns:
            tuple: Attack results depending on attack type
        """
        if self.attack_impl is None:
            self.initialize_models()
        return self.attack_impl.train(victim_model, num_queries, log_interval)

    def extract(self, query_data):
        """Extract predictions using trained attack model.
        
        Args:
            query_data: Input data for prediction
            
        Returns:
            torch.Tensor: Model predictions
        """
        if self.attack_impl is None:
            raise ValueError("Model must be trained before extraction")
        return self.attack_impl.extract(query_data)

    def evaluate(self, test_data):
        """Evaluate attack performance.
        
        Args:
            test_data: Test dataset
            
        Returns:
            dict: Dictionary containing evaluation metrics
        """
        if self.attack_impl is None:
            raise ValueError("Model must be trained before evaluation")
        return self.attack_impl.evaluate(test_data)
