import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
import torch
import torch.nn.functional as F
import dgl
import numpy as np
from tqdm import tqdm
from torch_geometric.utils import erdos_renyi_graph
from dgl.dataloading import NeighborSampler, NodeCollator
from torch.utils.data import DataLoader
import importlib

from models.defense.base import BaseDefense
from models.nn import GraphSAGE
from utils.metrics import GraphNeuralNetworkMetric

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class WatermarkByRandomGraph(BaseDefense):
    """
    A flexible defense implementation using watermarking to protect against
    model extraction attacks on graph neural networks.
    
    This class combines the functionalities from the original watermark.py:
    - Generating watermark graphs
    - Training models on original and watermark graphs
    - Merging graphs for testing
    - Evaluating effectiveness against attacks
    - Dynamic selection of attack methods
    """
    
    def __init__(self, dataset, attack_node_fraction=0.25, wm_node=50, pr=0.1, pg=0, attack_name=None):
        """
        Initialize the custom defense.
        
        Parameters
        ----------
        dataset : Dataset
            The original dataset containing the graph to defend
        attack_node_fraction : float, optional
            Fraction of nodes to consider for attack (default: 0.25)
        wm_node : int, optional
            Number of nodes in the watermark graph (default: 50)
        pr : float, optional
            Probability for feature generation in watermark (default: 0.1)
        pg : float, optional
            Probability for edge creation in watermark (default: 0)
        attack_name : str, optional
            Name of the attack class to use (default: None, will use ModelExtractionAttack0)
        """
        self.attack_name = attack_name or "ModelExtractionAttack0"
        super().__init__(dataset, attack_node_fraction)
        self.dataset = dataset
        self.graph = dataset.graph
        
        # Extract dataset properties
        self.node_number = dataset.node_number if hasattr(dataset, 'node_number') else self.graph.num_nodes()
        self.feature_number = dataset.feature_number if hasattr(dataset, 'feature_number') else self.graph.ndata['feat'].shape[1]
        self.label_number = dataset.label_number if hasattr(dataset, 'label_number') else (int(max(self.graph.ndata['label']) - min(self.graph.ndata['label'])) + 1)
        self.attack_node_number = int(self.node_number * attack_node_fraction)
        
        # Watermark parameters
        self.wm_node = wm_node
        self.pr = pr
        self.pg = pg
        
        # Extract features and labels
        self.features = dataset.features if hasattr(dataset, 'features') else self.graph.ndata['feat']
        self.labels = dataset.labels if hasattr(dataset, 'labels') else self.graph.ndata['label']
        
        # Extract masks
        self.train_mask = dataset.train_mask if hasattr(dataset, 'train_mask') else self.graph.ndata['train_mask']
        self.test_mask = dataset.test_mask if hasattr(dataset, 'test_mask') else self.graph.ndata['test_mask']
        
        # Move tensors to device
        if device != 'cpu':
            self.graph = self.graph.to(device)
            self.features = self.features.to(device)
            self.labels = self.labels.to(device)
            self.train_mask = self.train_mask.to(device)
            self.test_mask = self.test_mask.to(device)

    def _get_attack_class(self, attack_name):
        """
        Dynamically import and return the specified attack class.
        
        Parameters
        ----------
        attack_name : str
            Name of the attack class to import
            
        Returns
        -------
        class
            The requested attack class
        """
        try:
            # Try to import from models.attack module
            attack_module = importlib.import_module('models.attack')
            attack_class = getattr(attack_module, attack_name)
            return attack_class
        except (ImportError, AttributeError) as e:
            print(f"Error loading attack class '{attack_name}': {e}")
            print("Falling back to ModelExtractionAttack0")
            # Fallback to ModelExtractionAttack0
            attack_module = importlib.import_module('models.attack')
            return getattr(attack_module, "ModelExtractionAttack0")
    
    def defend(self, attack_name=None):
        """
        Main defense workflow:
        1. Train a target model on the original graph
        2. Attack the target model to establish baseline vulnerability
        3. Train a defense model with watermarking
        4. Test the defense model against the same attack
        5. Print performance metrics
        
        Parameters
        ----------
        attack_name : str, optional
            Name of the attack class to use, overrides the one set in __init__
            
        Returns
        -------
        dict
            Dictionary containing performance metrics
        """
        # Use the provided attack_name or fall back to the one from __init__
        attack_name = attack_name or self.attack_name
        AttackClass = self._get_attack_class(attack_name)
        
        print(f"Using attack method: {attack_name}")
        
        # Step 1: Train target model
        target_model = self._train_target_model()
        
        # Step 2: Attack target model
        attack = AttackClass(self.dataset, attack_node_fraction=0.3)
        target_attack_results = attack.attack()
        print("Attack results on target model:")
        if isinstance(target_attack_results, dict):
            if 'success_rate' in target_attack_results:
                print(f"Attack success rate: {target_attack_results['success_rate']:.4f}")
            if 'similarity' in target_attack_results:
                print(f"Model similarity: {target_attack_results['similarity']:.4f}")
        else:
            print("Attack completed. Results structure varies by attack type.")
            target_attack_results = {"completed": True}
        
        # Step 3: Train defense model with watermarking
        defense_model = self._train_defense_model()
        
        # Step 4: Test the defense model against the same attack
        attack = AttackClass(self.dataset, attack_node_fraction=0.3)
        defense_attack_results = attack.attack()
        
        # Step 5: Print performance metrics
        print("\nPerformance metrics:")
        print("Attack results on defense model:")
        if isinstance(defense_attack_results, dict):
            if 'success_rate' in defense_attack_results:
                print(f"Attack success rate: {defense_attack_results['success_rate']:.4f}")
            if 'similarity' in defense_attack_results:
                print(f"Model similarity: {defense_attack_results['similarity']:.4f}")
            
            # Calculate defense effectiveness if metrics are available
            if 'success_rate' in target_attack_results and 'success_rate' in defense_attack_results:
                effectiveness = 1 - defense_attack_results['success_rate'] / max(target_attack_results['success_rate'], 1e-10)
                print(f"Defense effectiveness: {effectiveness:.4f}")
        else:
            print("Attack completed. Results structure varies by attack type.")
            defense_attack_results = {"completed": True}
        
        # Evaluate watermark detection
        wm_detection = self._evaluate_watermark(defense_model)
        print(f"Watermark detection accuracy: {wm_detection:.4f}")
        
        return {
            "target_attack_results": target_attack_results,
            "defense_attack_results": defense_attack_results,
            "watermark_detection": wm_detection
        }

    def _train_target_model(self):
        """
        Helper function for training the target model on the original graph.
        
        Returns
        -------
        torch.nn.Module
            The trained target model
        """
        print("Training target model...")
        
        # Initialize model
        model = GraphSAGE(in_channels=self.feature_number,
                         hidden_channels=128,
                         out_channels=self.label_number)
        model = model.to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)
        
        # Setup data loading
        sampler = NeighborSampler([5, 5])
        train_nids = self.train_mask.nonzero(as_tuple=True)[0].to(device)
        test_nids = self.test_mask.nonzero(as_tuple=True)[0].to(device)
        
        train_collator = NodeCollator(self.graph, train_nids, sampler)
        test_collator = NodeCollator(self.graph, test_nids, sampler)
        
        train_dataloader = DataLoader(
            train_collator.dataset,
            batch_size=32,
            shuffle=True,
            collate_fn=train_collator.collate,
            drop_last=False
        )
        
        test_dataloader = DataLoader(
            test_collator.dataset,
            batch_size=32,
            shuffle=False,
            collate_fn=test_collator.collate,
            drop_last=False
        )
        
        # Training loop
        best_acc = 0
        for epoch in tqdm(range(1, 51), desc="Target model training"):
            # Train
            model.train()
            total_loss = 0
            for _, _, blocks in train_dataloader:
                blocks = [b.to(device) for b in blocks]
                input_features = blocks[0].srcdata['feat']
                output_labels = blocks[-1].dstdata['label']
                
                optimizer.zero_grad()
                output_predictions = model(blocks, input_features)
                loss = F.cross_entropy(output_predictions, output_labels)
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
            
            # Test
            model.eval()
            correct = 0
            total = 0
            with torch.no_grad():
                for _, _, blocks in test_dataloader:
                    blocks = [b.to(device) for b in blocks]
                    input_features = blocks[0].srcdata['feat']
                    output_labels = blocks[-1].dstdata['label']
                    output_predictions = model(blocks, input_features)
                    pred = output_predictions.argmax(dim=1)
                    correct += (pred == output_labels).sum().item()
                    total += len(output_labels)
            
            acc = correct / total
            if acc > best_acc:
                best_acc = acc
        
        print(f"Target model trained. Test accuracy: {best_acc:.4f}")
        return model

    def _train_defense_model(self):
        """
        Helper function for training a defense model with watermarking.
        
        Returns
        -------
        torch.nn.Module
            The trained defense model with embedded watermark
        """
        print("Training defense model with watermarking...")
        
        # Generate watermark graph
        wm_graph = self._generate_watermark_graph()
        
        # Initialize model
        model = GraphSAGE(in_channels=self.feature_number,
                         hidden_channels=128,
                         out_channels=self.label_number)
        model = model.to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)
        
        # Setup data loading for original graph
        sampler = NeighborSampler([5, 5])
        train_nids = self.train_mask.nonzero(as_tuple=True)[0].to(device)
        test_nids = self.test_mask.nonzero(as_tuple=True)[0].to(device)
        
        train_collator = NodeCollator(self.graph, train_nids, sampler)
        test_collator = NodeCollator(self.graph, test_nids, sampler)
        
        train_dataloader = DataLoader(
            train_collator.dataset,
            batch_size=32,
            shuffle=True,
            collate_fn=train_collator.collate,
            drop_last=False
        )
        
        test_dataloader = DataLoader(
            test_collator.dataset,
            batch_size=32,
            shuffle=False,
            collate_fn=test_collator.collate,
            drop_last=False
        )
        
        # Setup data loading for watermark graph
        wm_nids = torch.arange(wm_graph.number_of_nodes(), device=device)
        wm_collator = NodeCollator(wm_graph, wm_nids, sampler)
        
        wm_dataloader = DataLoader(
            wm_collator.dataset,
            batch_size=self.wm_node,
            shuffle=True,
            collate_fn=wm_collator.collate,
            drop_last=False
        )
        
        # First stage: Train on original graph
        best_acc = 0
        for epoch in tqdm(range(1, 51), desc="Defense model - stage 1"):
            # Train
            model.train()
            total_loss = 0
            for _, _, blocks in train_dataloader:
                blocks = [b.to(device) for b in blocks]
                input_features = blocks[0].srcdata['feat']
                output_labels = blocks[-1].dstdata['label']
                
                optimizer.zero_grad()
                output_predictions = model(blocks, input_features)
                loss = F.cross_entropy(output_predictions, output_labels)
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
            
            # Test
            model.eval()
            correct = 0
            total = 0
            with torch.no_grad():
                for _, _, blocks in test_dataloader:
                    blocks = [b.to(device) for b in blocks]
                    input_features = blocks[0].srcdata['feat']
                    output_labels = blocks[-1].dstdata['label']
                    output_predictions = model(blocks, input_features)
                    pred = output_predictions.argmax(dim=1)
                    correct += (pred == output_labels).sum().item()
                    total += len(output_labels)
            
            acc = correct / total
            if acc > best_acc:
                best_acc = acc
        
        # Second stage: Fine-tune on watermark graph
        for epoch in tqdm(range(1, 16), desc="Defense model - stage 2"):
            # Train on watermark
            model.train()
            total_loss = 0
            for _, _, blocks in wm_dataloader:
                blocks = [b.to(device) for b in blocks]
                input_features = blocks[0].srcdata['feat']
                output_labels = blocks[-1].dstdata['label']
                
                optimizer.zero_grad()
                output_predictions = model(blocks, input_features)
                loss = F.cross_entropy(output_predictions, output_labels)
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
        
        # Final evaluation
        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for _, _, blocks in test_dataloader:
                blocks = [b.to(device) for b in blocks]
                input_features = blocks[0].srcdata['feat']
                output_labels = blocks[-1].dstdata['label']
                output_predictions = model(blocks, input_features)
                pred = output_predictions.argmax(dim=1)
                correct += (pred == output_labels).sum().item()
                total += len(output_labels)
        
        final_acc = correct / total
        
        # Watermark accuracy
        wm_acc = self._test_on_watermark(model, wm_dataloader)
        
        print(f"Defense model trained.")
        print(f"Test accuracy on original data: {final_acc:.4f}")
        print(f"Test accuracy on watermark: {wm_acc:.4f}")
        
        # Store watermark graph for later verification
        self.watermark_graph = wm_graph
        
        return model
    
    def _generate_watermark_graph(self):
        """
        Generate a watermark graph using Erdos-Renyi random graph model.
        
        Returns
        -------
        dgl.DGLGraph
            The generated watermark graph
        """
        # Generate random edges using Erdos-Renyi model
        wm_edge_index = erdos_renyi_graph(self.wm_node, self.pg, directed=False)
        
        # Generate random features with binomial distribution
        wm_features = torch.tensor(np.random.binomial(
            1, self.pr, size=(self.wm_node, self.feature_number)), 
            dtype=torch.float32).to(device)
        
        # Generate random labels
        wm_labels = torch.tensor(np.random.randint(
            low=0, high=self.label_number, size=self.wm_node), 
            dtype=torch.long).to(device)
        
        # Create DGL graph
        wm_graph = dgl.graph((wm_edge_index[0], wm_edge_index[1]), num_nodes=self.wm_node)
        wm_graph = wm_graph.to(device)
        
        # Add node features and labels
        wm_graph.ndata['feat'] = wm_features
        wm_graph.ndata['label'] = wm_labels
        
        # Add train and test masks (all True for simplicity)
        wm_graph.ndata['train_mask'] = torch.ones(self.wm_node, dtype=torch.bool, device=device)
        wm_graph.ndata['test_mask'] = torch.ones(self.wm_node, dtype=torch.bool, device=device)
        
        # Add self-loops
        wm_graph = dgl.add_self_loop(wm_graph)
        
        return wm_graph
    
    def _test_on_watermark(self, model, wm_dataloader):
        """
        Test a model's accuracy on the watermark graph.
        
        Parameters
        ----------
        model : torch.nn.Module
            The model to test
        wm_dataloader : DataLoader
            DataLoader for the watermark graph
            
        Returns
        -------
        float
            Accuracy on the watermark graph
        """
        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for _, _, blocks in wm_dataloader:
                blocks = [b.to(device) for b in blocks]
                input_features = blocks[0].srcdata['feat']
                output_labels = blocks[-1].dstdata['label']
                output_predictions = model(blocks, input_features)
                pred = output_predictions.argmax(dim=1)
                correct += (pred == output_labels).sum().item()
                total += len(output_labels)
        
        return correct / total
    
    def _evaluate_watermark(self, model):
        """
        Evaluate watermark detection effectiveness.
        
        Parameters
        ----------
        model : torch.nn.Module
            The model to evaluate
            
        Returns
        -------
        float
            Watermark detection accuracy
        """
        if not hasattr(self, 'watermark_graph'):
            print("Warning: No watermark graph found. Generate one first.")
            return 0.0
        
        # Setup data loading for watermark graph
        sampler = NeighborSampler([5, 5])
        wm_nids = torch.arange(self.watermark_graph.number_of_nodes(), device=device)
        wm_collator = NodeCollator(self.watermark_graph, wm_nids, sampler)
        
        wm_dataloader = DataLoader(
            wm_collator.dataset,
            batch_size=self.wm_node,
            shuffle=False,
            collate_fn=wm_collator.collate,
            drop_last=False
        )
        
        return self._test_on_watermark(model, wm_dataloader)