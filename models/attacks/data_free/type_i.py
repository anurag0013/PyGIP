from .base import DataFreeBaseAttack
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
from .generator import GraphGenerator
from .surrogate import SurrogateModel

class TypeIAttack(DataFreeBaseAttack):
    """Type I data-free attack implementation using zeroth-order optimization.
    
    This attack uses a generator to create synthetic graph data and trains a
    surrogate model to mimic the victim model's behavior using zeroth-order
    optimization techniques.
    
    Attributes:
        generator: Graph generator model
        surrogate_model: Surrogate model for attack
        victim_model: Target victim model
        device: Computing device (CPU/GPU)
    """
    
    def __init__(self, noise_dim, num_nodes, feature_dim,
                 generator_lr=1e-6, surrogate_lr=0.001,
                 n_generator_steps=2, n_surrogate_steps=5):
        """Initialize Type I attack.
        
        Args:
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

    def initialize_models(self):
        """Initialize generator and surrogate models."""
        self.generator = GraphGenerator(
            self.noise_dim, 
            self.num_nodes, 
            self.feature_dim
        ).to(self.device)
        
        self.surrogate_model = SurrogateModel(
            self.feature_dim,
            64,  # hidden dimension
            self.victim_model.output_dim
        ).to(self.device)
        
        self.generator_optimizer = optim.Adam(
            self.generator.parameters(), 
            lr=self.generator_lr
        )
        self.surrogate_optimizer = optim.Adam(
            self.surrogate_model.parameters(), 
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
        """Train the generator model.
        
        Returns:
            float: Average generator loss
        """
        self.generator.train()
        self.surrogate_model.eval()

        total_loss = 0
        for _ in range(self.n_generator_steps):
            self.generator_optimizer.zero_grad()
            
            features, edge_index = self.generate_graph()

            with torch.no_grad():
                victim_output = self.victim_model(features, edge_index)
            surrogate_output = self.surrogate_model(features, edge_index)

            loss = -self.criterion(surrogate_output, victim_output.argmax(dim=1))

            # Zeroth-order optimization with multiple random directions
            epsilon = 1e-6
            num_directions = 2
            estimated_gradient = torch.zeros_like(features)
            
            for _ in range(num_directions):
                u = torch.randn_like(features)
                perturbed_features = features + epsilon * u
                
                with torch.no_grad():
                    perturbed_victim_output = self.victim_model(perturbed_features, edge_index)
                perturbed_surrogate_output = self.surrogate_model(perturbed_features, edge_index)
                perturbed_loss = -self.criterion(perturbed_surrogate_output, 
                                               perturbed_victim_output.argmax(dim=1))
                
                estimated_gradient += (perturbed_loss - loss) / epsilon * u
            
            estimated_gradient /= num_directions
            features.grad = estimated_gradient

            self.generator_optimizer.step()
            total_loss += loss.item()

        return total_loss / self.n_generator_steps

    def train_surrogate(self):
        """Train the surrogate model.
        
        Returns:
            float: Average surrogate loss
        """
        self.generator.eval()
        self.surrogate_model.train()

        total_loss = 0
        for _ in range(self.n_surrogate_steps):
            self.surrogate_optimizer.zero_grad()
            
            features, edge_index = self.generate_graph()

            with torch.no_grad():
                victim_output = self.victim_model(features, edge_index)
            surrogate_output = self.surrogate_model(features, edge_index)

            loss = self.criterion(surrogate_output, victim_output.argmax(dim=1))
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.surrogate_model.parameters(), max_norm=1.0)
            self.surrogate_optimizer.step()

            total_loss += loss.item()

        return total_loss / self.n_surrogate_steps

    def train(self, victim_model, num_queries=300, log_interval=10):
        """Train the attack model.
        
        Args:
            victim_model: The target model to attack
            num_queries (int): Number of queries to make
            log_interval (int): Interval for logging progress
            
        Returns:
            tuple: (surrogate_model, generator_losses, surrogate_losses)
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

        return self.surrogate_model, generator_losses, surrogate_losses
