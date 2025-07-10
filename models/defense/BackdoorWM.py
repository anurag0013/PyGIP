import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
import torch
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score
import random

from models.defense.base import BaseDefense
from utils.metrics import GraphNeuralNetworkMetric
from models.nn import GCN

# Use device from base class
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class BackdoorWM(BaseDefense):
    def __init__(self, dataset, attack_node_fraction, model_path=None, trigger_rate=0.01, l=20, target_label=0):
        self.trigger_rate = trigger_rate
        self.l = l
        self.target_label = target_label
        super().__init__(dataset, attack_node_fraction)

    def _load_model(self, model_path):
        """
        Load a pre-trained model.
        """
        from models.nn import GCN

        # Create the model
        self.net1 = GCN(self.feature_number, self.label_number).to(device)

        # Load the saved state dict
        self.net1.load_state_dict(torch.load(model_path, map_location=device))

        # Set to evaluation mode
        self.net1.eval()

    def inject_backdoor_trigger(self, data, trigger_rate=None, trigger_feat_val=0.99, l=None, target_label=None):
        """Feature-based Trigger Injection"""
        if trigger_rate is None:
            trigger_rate = self.trigger_rate
        if l is None:
            l = self.l
        if target_label is None:
            target_label = self.target_label

        num_nodes = data.shape[0]
        num_feats = data.shape[1]
        num_trigger_nodes = int(trigger_rate * num_nodes)

        trigger_nodes = random.sample(range(num_nodes), num_trigger_nodes)
        for node in trigger_nodes:
            feature_indices = random.sample(range(num_feats), l)
            data[node][feature_indices] = trigger_feat_val
        return data, trigger_nodes

    def train_target_model(self):
        """
        Train the target model with backdoor injection.
        """
        # Initialize GNN model
        self.net1 = GCN(self.feature_number, self.label_number).to(device)
        optimizer = torch.optim.Adam(self.net1.parameters(), lr=0.01, weight_decay=5e-4)

        # Inject backdoor trigger
        poisoned_features = self.features.clone()
        poisoned_labels = self.labels.clone()

        poisoned_features_cpu = poisoned_features.cpu()
        poisoned_features_cpu, trigger_nodes = self.inject_backdoor_trigger(
            poisoned_features_cpu,
            trigger_rate=self.trigger_rate,
            l=self.l,
            target_label=self.target_label
        )
        poisoned_features = poisoned_features_cpu.to(device)

        # Modify labels for trigger nodes
        for node in trigger_nodes:
            poisoned_labels[node] = self.target_label

        self.trigger_nodes = trigger_nodes
        self.poisoned_features = poisoned_features
        self.poisoned_labels = poisoned_labels

        # Training loop
        for epoch in range(200):
            self.net1.train()

            # Forward pass
            logits = self.net1(self.graph, poisoned_features)
            logp = F.log_softmax(logits, dim=1)
            loss = F.nll_loss(logp[self.train_mask], poisoned_labels[self.train_mask])

            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # Validation (optional)
            if epoch % 50 == 0:
                self.net1.eval()
                with torch.no_grad():
                    logits_val = self.net1(self.graph, poisoned_features)
                    logp_val = F.log_softmax(logits_val, dim=1)
                    pred = logp_val.argmax(dim=1)
                    acc_val = (pred[self.test_mask] == poisoned_labels[self.test_mask]).float().mean()
                    print(f"  Epoch {epoch}: training... Validation Accuracy: {acc_val.item():.4f}")

        return self.net1

    def verify_backdoor(self, model, trigger_nodes, target_label):
        """Verify backdoor attack success rate"""
        model.eval()
        with torch.no_grad():
            out = model(self.graph, self.poisoned_features)
            pred = out.argmax(dim=1)
            correct = (pred[trigger_nodes] == target_label).sum().item()
            return correct / len(trigger_nodes)

    def evaluate_model(self, model, features, labels):
        """Evaluate model performance"""
        model.eval()
        with torch.no_grad():
            out = model(self.graph, features)
            logits = out[self.test_mask]
            preds = logits.argmax(dim=1).cpu()
            labels_test = labels[self.test_mask].cpu()
            probs = F.softmax(logits, dim=1).cpu()

            return {
                'accuracy': accuracy_score(labels_test, preds),
                'f1': f1_score(labels_test, preds, average='macro'),
                'precision': precision_score(labels_test, preds, average='macro'),
                'recall': recall_score(labels_test, preds, average='macro'),
                'auroc': roc_auc_score(labels_test, probs, multi_class='ovo')
            }

    def defend(self):
        """
        Execute the backdoor watermark attack.
        """
        print("=========Backdoor Watermark Attack==========================")

        # If model wasn't trained yet, train it
        if not hasattr(self, 'net1'):
            self.train_target_model()

        # Evaluate the backdoored model
        metrics = self.evaluate_model(self.net1, self.poisoned_features, self.poisoned_labels)
        wm_acc = self.verify_backdoor(self.net1, self.trigger_nodes, self.target_label)

        # Create performance metrics object
        performance_metrics = GraphNeuralNetworkMetric()
        performance_metrics.accuracy = metrics['accuracy']
        performance_metrics.f1 = metrics['f1']
        performance_metrics.fidelity = wm_acc  # Use watermark accuracy as fidelity

        print("========================Final results:=========================================")
        print(f"Model Accuracy: {metrics['accuracy']:.4f}")
        print(f"F1 Score: {metrics['f1']:.4f}")
        print(f"Watermark Accuracy: {wm_acc:.4f}")
        print(performance_metrics)

        return performance_metrics, self.net1
