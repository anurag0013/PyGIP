from .base import DataFreeBaseAttack
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
from .generator import GraphGenerator
from .surrogate import SurrogateModel

class TypeIIIAttack(DataFreeBaseAttack):
    """Type III data-free attack implementation using model ensemble.
    
    This attack uses two surrogate models and maximizes their disagreement
    to generate more informative synthetic graph data. The generator is 
    trained to create samples where the surrogate models disagree.
    
    Attributes:
        generator: Graph generator model
        surrogate_model1: First surrogate model
        surrogate_model2: Second surrogate model
        victim_model: Target victim model
        device: Computing device (CPU/GPU)
    """
    
    def __init__(self, noise_dim, num_nodes, feature_dim,
                 generator_lr=1e-6, surrogate_lr=0.001,
                 n_generator_steps=2, n_surrogate_steps=5):
        """Initialize Type III attack.
        
        Args:
            noise_dim (int): Dimension of input noise vector
            num_nodes (int): Number of nodes in generated graph
            feature_dim (int): Dimension of node features
            generator_lr (float): Learning rate for generator
            surrogate_lr (float): Learning rate for surrogate models
            n_generator_steps (int): Number of generator training steps
            n_surrogate_steps (int): Number of surrogate training steps
        """
        super().__init__(noise_dim, num_nodes, feature_dim,
                        generator_lr, surrogate_lr,
                        n_generator_steps, n_surrogate_steps)
        self.surrogate_model1 = None
        self.surrogate_model2 = None
        self.surrogate_optimizer1 = None
        self.surrogate_optimizer2 = None

    def initialize_models(self):
        """Initialize generator and two surrogate models."""
        self.generator = GraphGenerator(
            self.noise_dim, 
            self.num_nodes, 
            self.feature_dim
        ).to(self.device)
        
        # Initialize two surrogate models
        self.surrogate_model1 = SurrogateModel(
            self.feature_dim,
            64,  # hidden dimension
            self.victim_model.output_dim
        ).to(self.device)
        
        self.surrogate_model2 = SurrogateModel(
            self.feature_dim,
            64,  # hidden dimension
            self.victim_model.output_dim
        ).to(self.device)
        
        # Initialize optimizers
        self.generator_optimizer = optim.Adam(
            self.generator.parameters(), 
            lr=self.generator_lr
        )
        self.surrogate_optimizer1 = optim.Adam(
            self.surrogate_model1.parameters(), 
            lr=self.surrogate_lr
        )
        self.surrogate_optimizer2 = optim.Adam(
            self.surrogate_model2.parameters(), 
            lr=self.surrogate_lr
        )

    def generate_graph(self):
        """Generate synthetic graph data.
        
        Returns:
            tuple: (features, edge_index)
        """
        z = torch.randn(1, self.noise_dim).to(self.device)
        features, adj = self.generator(z)
        edge_index = self.generator.adj_to_edge_index(adj)
        return features, edge_index

    def train_generator(self):
        """Train the generator to maximize surrogate models' disagreement.
        
        Returns:
            float: Average generator loss
        """
        self.generator.train()
        self.surrogate_model1.eval()
        self.surrogate_model2.eval()

        total_loss = 0
        for _ in range(self.n_generator_steps):
            self.generator_optimizer.zero_grad()
            
            features, edge_index = self.generate_graph()

            surrogate_output1 = self.surrogate_model1(features, edge_index)
            surrogate_output2 = self.surrogate_model2(features, edge_index)

            # Maximize disagreement between surrogate models
            loss = -torch.mean(torch.std(torch.stack([surrogate_output1, surrogate_output2]), dim=0))
            loss.backward()

            self.generator_optimizer.step()
            total_loss += loss.item()

        return total_loss / self.n_generator_steps

    def train_surrogate(self):
        """Train both surrogate models to mimic victim model behavior.
        
        Returns:
            float: Average surrogate loss
        """
        self.generator.eval()
        self.surrogate_model1.train()
        self.surrogate_model2.train()

        total_loss = 0
        for _ in range(self.n_surrogate_steps):
            self.surrogate_optimizer1.zero_grad()
            self.surrogate_optimizer2.zero_grad()
            
            features, edge_index = self.generate_graph()

            with torch.no_grad():
                victim_output = self.victim_model(features, edge_index)
            surrogate_output1 = self.surrogate_model1(features, edge_index)
            surrogate_output2 = self.surrogate_model2(features, edge_index)

            loss1 = self.criterion(surrogate_output1, victim_output.argmax(dim=1))
            loss2 = self.criterion(surrogate_output2, victim_output.argmax(dim=1))
            
            # Combine losses and backpropagate
            combined_loss = loss1 + loss2
            combined_loss.backward()
            
            torch.nn.utils.clip_grad_norm_(self.surrogate_model1.parameters(), max_norm=1.0)
            torch.nn.utils.clip_grad_norm_(self.surrogate_model2.parameters(), max_norm=1.0)
            
            self.surrogate_optimizer1.step()
            self.surrogate_optimizer2.step()

            total_loss += combined_loss.item() / 2

        return total_loss / self.n_surrogate_steps

    def train(self, victim_model, num_queries=300, log_interval=10):
        """Train the attack models.
        
        Args:
            victim_model: The target model to attack
            num_queries (int): Number of queries to make
            log_interval (int): Interval for logging progress
            
        Returns:
            tuple: ((surrogate_model1, surrogate_model2), generator_losses, surrogate_losses)
        """
        self.victim_model = victim_model
        self.initialize_models()
        
        generator_losses = []
        surrogate_losses = []

        pbar = tqdm(range(num_queries), desc="Attacking")
        for query in pbar:
            gen_loss = self.train_generator()
            surr_loss = self.train_surrogate()

            generator_losses.append(gen_loss)
            surrogate_losses.append(surr_loss)

            if (query + 1) % log_interval == 0:
                pbar.set_postfix({
                    'Gen Loss': f"{gen_loss:.4f}",
                    'Surr Loss': f"{surr_loss:.4f}"
                })

        return (self.surrogate_model1, self.surrogate_model2), generator_losses, surrogate_losses

    def extract(self, query_data):
        """Extract predictions using ensemble of surrogate models.
        
        Args:
            query_data: Input data for prediction
            
        Returns:
            torch.Tensor: Averaged predictions from both surrogate models
        """
        self.surrogate_model1.eval()
        self.surrogate_model2.eval()
        with torch.no_grad():
            pred1 = self.surrogate_model1(query_data.x, query_data.edge_index)
            pred2 = self.surrogate_model2(query_data.x, query_data.edge_index)
            # Average predictions from both models
            return (pred1 + pred2) / 2

    def evaluate(self, test_data):
        """Evaluate attack performance using ensemble predictions.
        
        Args:
            test_data: Test dataset
            
        Returns:
            dict: Dictionary containing evaluation metrics
        """
        pred = self.extract(test_data)
        acc = (pred.argmax(dim=1) == test_data.y).float().mean()
        return {'accuracy': acc.item()}
