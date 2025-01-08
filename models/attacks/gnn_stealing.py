from ..base.attack import BaseAttack
from ...utils.metrics import GraphNeuralNetworkMetric
from ...utils.models import Gcn_Net

import torch
import torch.nn.functional as F
from sklearn.manifold import TSNE
from GNNSTEALING.code.src.utils import split_graph_different_ratio, delete_dgl_graph_edge, split_graph
from GNNSTEALING.code.src.gin import GIN, evaluate_gin_target, run_gin_target
from GNNSTEALING.code.src.gat import evaluate_gat_target, run_gat_target
from GNNSTEALING.code.src.sage import evaluate_sage_target, run_sage_target
from GNNSTEALING.code.src.ginsurrogate import run_gin_surrogate, evaluate_gin_surrogate
from GNNSTEALING.code.src.gatsurrogate import run_gat_surrogate, evaluate_gat_surrogate
from GNNSTEALING.code.src.sagesurrogate import run_sage_surrogate, evaluate_sage_surrogate
from GNNSTEALING.code.core.model_handler import ModelHandler
from scipy import sparse
import dgl
import numpy as np

class GNNStealing(BaseAttack):

    def __init__(self,
                 dataset,
                 attack_node_fraction,
                 target_model_type,
                 surrogate_model_type, 
                 recovery_method,
                 structure='original',
                 delete_edges='no',
                 transform='TSNE',
                 device='cuda' if torch.cuda.is_available() else 'cpu'):
        super().__init__(dataset, attack_node_fraction)
        self.target_model_type = target_model_type
        self.surrogate_model_type = surrogate_model_type
        self.recovery_method = recovery_method
        self.structure = structure
        self.delete_edges = delete_edges
        self.transform = transform
        self.device = device

        self.n_classes = self.label_number if hasattr(dataset, 'num_classes') else 6

        # Split the graph
        self.train_g, self.val_g, self.test_g = split_graph(
            self.graph, frac_list=[0.6, 0.2, 0.2])

        # Train target model
        self.target_model = self.train_target_model_stl()

def train_target_model_stl(self):
        print(f"Entering train_target_model, target_model_type: {self.target_model_type}")
        print("\nModel information:")
        self.preprocess_dataset()

        args = self._get_default_args()

        print(f"Dataset: {args.dataset}")
        print(f"Number of nodes: {args.num_nodes}")
        print(f"Number of features: {args.in_feats}")
        print(f"Number of classes: {args.n_classes}")
        print(f"Unique labels: {torch.unique(self.labels)}")

        if self.target_model_type == "gat":
            data = self.train_g, self.val_g, self.test_g, args.in_feats, self.labels, args.n_classes, self.graph, args.head
            target_model = run_gat_target(args, self.device, data)
        elif self.target_model_type == "gin":
            data = self.train_g, self.val_g, self.test_g, args.in_feats, self.labels, args.n_classes
            target_model = run_gin_target(args, self.device, data)
        elif self.target_model_type == "sage":
            data = args.in_feats, args.n_classes, self.train_g, self.val_g, self.test_g
            target_model = run_sage_target(args, self.device, data)
        else:
            raise ValueError("target_model_type should be gat, gin, or sage")

        return target_model

    def preprocess_dataset(self):
        if not hasattr(self, 'train_mask'):
            num_nodes = self.node_number
            self.train_mask = torch.zeros(num_nodes, dtype=torch.bool)
            self.val_mask = torch.zeros(num_nodes, dtype=torch.bool)
            self.test_mask = torch.zeros(num_nodes, dtype=torch.bool)
            
            indices = torch.randperm(num_nodes)
            self.train_mask[indices[:int(num_nodes * 0.6)]] = True
            self.val_mask[indices[int(num_nodes * 0.6):int(num_nodes * 0.8)]] = True
            self.test_mask[indices[int(num_nodes * 0.8):]] = True

        if 'feat' in self.graph.ndata:
            self.features = self.graph.ndata['feat']
        elif 'attr' in self.graph.ndata:
            self.features = self.graph.ndata['attr']
        else:
            raise ValueError("No node features found in the graph.")

        if 'label' not in self.graph.ndata:
            raise ValueError("No node labels found in the graph.")
        self.labels = self.graph.ndata['label']

    def _get_default_args(self):
        class Args:
            def __init__(self, target_model_type, dataset):
                self.num_epochs = 200
                self.num_hidden = 256
                self.num_layers = 3
                self.fan_out = '10,25' if target_model_type == 'sage' else '10,10,10'
                self.batch_size = 512
                self.val_batch_size = 512
                self.log_every = 20
                self.eval_every = 100
                self.lr = 0.001
                self.dropout = 0.5
                self.num_workers = 0
                self.inductive = True if target_model_type == 'sage' else False
                self.head = 4
                self.wd = 0
                self.dataset = dataset.__class__.__name__
                
                # Use the attributes from the dataset
                self.in_feats = dataset.feature_number
                self.n_classes = dataset.label_number
                self.num_nodes = dataset.node_number
            

        return Args(self.target_model_type, self.dataset)
    
    def idgl_reconstruction(self, train_g):
        config = {
            'dgl_graph': train_g,
            'cuda_id': self.device.index if self.device.type == 'cuda' else -1,
            # Add other necessary configuration parameters for IDGL
        }
        
        model = ModelHandler(config)
        model.train()
        _, adj = model.test()
        
        adj = adj.clone().detach().cpu().numpy()
        
        # Thresholding based on dataset
        if self.dataset.__class__.__name__.lower() in ['acm', 'amazon_cs']:
            adj = (adj > 0.9).astype(np.int)
        elif self.dataset.__class__.__name__.lower() in ['coauthor_phy']:
            adj = (adj >= 0.999).astype(np.int)
        else:
            adj = (adj > 0.999).astype(np.int)

        sparse_adj = sparse.csr_matrix(adj)
        G_QUERY = dgl.from_scipy(sparse_adj)
        G_QUERY.ndata['features'] = train_g.ndata['features']
        G_QUERY.ndata['labels'] = train_g.ndata['labels']
        
        return G_QUERY

    def generate_query_graph(self):
        self.preprocess_dataset()
        train_g, self.val_g, self.test_g = split_graph_different_ratio(self.graph, frac_list=[0.3, 0.2, 0.5], ratio=1.0)
        
        if self.structure == 'original':
            G_QUERY = train_g
            if self.delete_edges == "yes":
                G_QUERY = delete_dgl_graph_edge(train_g)
        elif self.structure == 'idgl':
            G_QUERY = self.idgl_reconstruction(train_g)
        else:
            raise ValueError("Invalid structure parameter")

        # Add self-loops to the graph
        G_QUERY = dgl.add_self_loop(G_QUERY)
        self.val_g = dgl.add_self_loop(self.val_g)
        self.test_g = dgl.add_self_loop(self.test_g)

        return G_QUERY

    def query_target_model(self, G_QUERY):
        # Determine the correct feature key
        feature_key = 'features' if 'features' in G_QUERY.ndata else 'feat' if 'feat' in G_QUERY.ndata else 'attr' if 'attr' in G_QUERY.ndata else None
        if feature_key is None:
            raise ValueError("No node features found in the graph. Expected 'features', 'feat', or 'attr'.")

        # Determine the correct label key
        label_key = 'labels' if 'labels' in G_QUERY.ndata else 'label'

        features = G_QUERY.ndata[feature_key]
        labels = G_QUERY.ndata[label_key]

        if self.target_model_type == 'sage':
            query_acc, query_preds, query_embs = evaluate_sage_target(
                self.target_model, G_QUERY, features, labels, G_QUERY.nodes(), 1000, self.device)
        elif self.target_model_type == 'gin':
            query_acc, query_preds, query_embs = evaluate_gin_target(
                self.target_model, G_QUERY, features, labels, G_QUERY.nodes(), 1000, self.device)
        elif self.target_model_type == 'gat':
            query_acc, query_preds, query_embs = evaluate_gat_target(
                self.target_model, G_QUERY, features, labels, G_QUERY.nodes(), 1000, 4, self.device)
        else:
            raise ValueError("Invalid target model type")

        return query_acc, query_preds.to(self.device), query_embs.to(self.device)
    def preprocess_query_response(self, G_QUERY, query_preds, query_embs):
        if self.recovery_method == 'prediction':
            data = G_QUERY.ndata['features'].shape[1], query_preds.shape[1], G_QUERY, self.val_g, self.test_g, query_preds
        elif self.recovery_method == 'embedding':
            data = G_QUERY.ndata['features'].shape[1], query_preds.shape[1], G_QUERY, self.val_g, self.test_g, query_embs
        elif self.recovery_method == 'projection':
            tsne_embs = TSNE(n_components=2).fit_transform(query_embs.cpu().numpy())
            tsne_embs = torch.from_numpy(tsne_embs).float().to(self.device)
            data = G_QUERY.ndata['features'].shape[1], query_preds.shape[1], G_QUERY, self.val_g, self.test_g, tsne_embs
        else:
            raise ValueError("Invalid recovery method")

        return data

    def train_surrogate_model(self, data):
        if self.surrogate_model_type == 'gin':
            model_s, classifier, detached_classifier = run_gin_surrogate(
                self.device, data,
                fan_out='10,25', 
                batch_size=1000,
                num_workers=0,
                num_hidden=256,
                num_layers=2,
                dropout=0.5,
                lr=0.005,
                num_epochs=100,
                log_every=20,
                eval_every=5
            )
        elif self.surrogate_model_type == 'gat':
            model_s, classifier, detached_classifier = run_gat_surrogate(
                self.device, data,
                fan_out='10,25',
                batch_size=1000,
                num_workers=0,
                num_hidden=256,
                num_layers=2,
                head=4,  # Added this parameter for GAT
                dropout=0.5,
                lr=0.005,
                num_epochs=100,
                log_every=20,
                eval_every=5
            )
        elif self.surrogate_model_type == 'sage':
            model_s, classifier, detached_classifier = run_sage_surrogate(
                self.device, data,
                fan_out='10,25',
                batch_size=1000,
                num_workers=0,
                num_hidden=256,
                num_layers=2,
                dropout=0.5,
                lr=0.005,
                num_epochs=100,
                log_every=20,
                eval_every=5
            )
        else:
            raise ValueError("Invalid surrogate model type")

        return model_s, classifier, detached_classifier 


    def evaluate_surrogate_model(self, model_s, classifier, test_g):
        if self.surrogate_model_type == 'gin':
            acc_surrogate, preds_surrogate, embds_surrogate = evaluate_gin_surrogate(model_s, classifier, test_g, 
                                                                                     test_g.ndata['features'], test_g.ndata['labels'],
                                                                                     test_g.nodes(), 1000, self.device)
        elif self.surrogate_model_type == 'gat':
            acc_surrogate, preds_surrogate, embds_surrogate = evaluate_gat_surrogate(model_s, classifier, test_g,
                                                                                     test_g.ndata['features'], test_g.ndata['labels'],
                                                                                     test_g.nodes(), 1000, 4, self.device)
        elif self.surrogate_model_type == 'sage':
            acc_surrogate, preds_surrogate, embds_surrogate = evaluate_sage_surrogate(model_s, classifier, test_g,
                                                                                      test_g.ndata['features'], test_g.ndata['labels'],
                                                                                      test_g.nodes(), 1000, self.device)
        else:
            raise ValueError("Invalid surrogate model type")

        return acc_surrogate, preds_surrogate, embds_surrogate

    def compute_fidelity(self, preds_surrogate, preds_target):
        return (torch.argmax(preds_surrogate, dim=1) == torch.argmax(preds_target, dim=1)).float().mean().item()


    def attack(self):
        """执行攻击"""
        G_QUERY = self.generate_query_graph()
        query_acc, query_preds, query_embs = self.query_target_model(G_QUERY)
        data = self.preprocess_query_response(G_QUERY, query_preds, query_embs)
        model_s, classifier, detached_classifier = self.train_surrogate_model(data)
        acc_surrogate, preds_surrogate, embds_surrogate = self.evaluate_surrogate_model(
            model_s, classifier, self.test_g)
        _, preds_target, _ = self.query_target_model(self.test_g)
        fidelity = self.compute_fidelity(preds_surrogate, preds_target)
        accuracy = detached_classifier.score(
            embds_surrogate.clone().detach().cpu().numpy(),
            self.test_g.ndata['labels'])

        print("Attack Results:")
        print(f"Surrogate Model Accuracy: {acc_surrogate:.4f}")
        print(f"Target Model Accuracy: {query_acc:.4f}") 
        print(f"Accuracy: {accuracy:.4f}")
        print(f"Fidelity: {fidelity:.4f}")