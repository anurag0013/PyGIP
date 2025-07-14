import dgl
import numpy as np
import torch
from dgl import DGLGraph
from dgl.data import AmazonCoBuyComputerDataset  # Amazon-Computer
from dgl.data import AmazonCoBuyPhotoDataset  # Amazon-Photo
from dgl.data import citation_graph  # Cora, CiteSeer, PubMed
from sklearn.model_selection import StratifiedShuffleSplit
from torch_geometric.data import Data as PyGData
from torch_geometric.datasets import Amazon  # Amazon Computers, Photo
from torch_geometric.datasets import Planetoid  # Cora, CiteSeer, PubMed
from torch_geometric.datasets import TUDataset  # ENZYMES


def dgl_to_tg(dgl_graph):
    edge_index = torch.stack(dgl_graph.edges())
    x = dgl_graph.ndata.get('feat')
    y = dgl_graph.ndata.get('label')

    train_mask = dgl_graph.ndata.get('train_mask')
    val_mask = dgl_graph.ndata.get('val_mask')
    test_mask = dgl_graph.ndata.get('test_mask')

    data = PyGData(x=x, edge_index=edge_index, y=y,
                   train_mask=train_mask, val_mask=val_mask, test_mask=test_mask)
    return data


def tg_to_dgl(py_g_data):
    edge_index = py_g_data.edge_index
    dgl_graph = dgl.graph((edge_index[0], edge_index[1]))

    if py_g_data.x is not None:
        dgl_graph.ndata['feat'] = py_g_data.x
    if py_g_data.y is not None:
        dgl_graph.ndata['label'] = py_g_data.y

    if hasattr(py_g_data, 'train_mask') and py_g_data.train_mask is not None:
        dgl_graph.ndata['train_mask'] = py_g_data.train_mask
    if hasattr(py_g_data, 'val_mask') and py_g_data.val_mask is not None:
        dgl_graph.ndata['val_mask'] = py_g_data.val_mask
    if hasattr(py_g_data, 'test_mask') and py_g_data.test_mask is not None:
        dgl_graph.ndata['test_mask'] = py_g_data.test_mask

    return dgl_graph


class Dataset(object):
    def __init__(self, api_type='dgl', path='./data'):
        assert api_type in {'dgl', 'pyg'}, 'API type must be dgl or pyg'
        self.api_type = api_type
        self.path = path
        self.dataset_name = self.get_name()

        # DGLGraph or PyGData
        self.graph_dataset = None
        self.graph_data = None

        # meta data
        self.num_nodes = 0
        self.num_features = 0
        self.num_classes = 0

    def get_name(self):
        return self.__class__.__name__

    def load_dgl_data(self):
        raise NotImplementedError("load_dgl_data not implemented in subclasses.")

    def load_pyg_data(self):
        raise NotImplementedError("load_pyg_data not implemented in subclasses.")

    def _load_meta_data(self):
        if isinstance(self.graph_data, DGLGraph):
            self.num_nodes = self.graph_data.number_of_nodes()
            self.num_features = len(self.graph_data.ndata['feat'][0])
            self.num_classes = int(max(self.graph_data.ndata['label']) - min(self.graph_data.ndata['label'])) + 1
        elif isinstance(self.graph_data, PyGData):
            self.num_nodes = self.graph_data.num_nodes
            self.num_features = self.graph_dataset.num_node_features
            self.num_classes = self.graph_dataset.num_classes
        else:
            raise TypeError("graph_data must be either DGLGraph or torch_geometric.data.Data.")

    def _generate_train_test_masks(self, train_ratio=0.8):
        if self.graph_data is None:
            raise ValueError("graph_data is not loaded.")

        try:
            import dgl
        except ImportError:
            dgl = None

        try:
            from torch_geometric.data import Data
        except ImportError:
            Data = None

        is_dgl = dgl and isinstance(self.graph_data, dgl.DGLGraph)
        is_pyg = Data and isinstance(self.graph_data, Data)

        if not (is_dgl or is_pyg):
            raise TypeError("graph_data must be either DGLGraph or torch_geometric.data.Data.")

        # Check if masks already exist
        if is_dgl:
            if all(k in self.graph_data.ndata for k in ['train_mask', 'val_mask', 'test_mask']):
                print("Masks already exist in DGL graph. Skipping mask generation.")
                return
            num_nodes = self.graph_data.num_nodes()
        else:  # PyG
            if all(hasattr(self.graph_data, k) for k in ['train_mask', 'val_mask', 'test_mask']):
                print("Masks already exist in PyG data. Skipping mask generation.")
                return
            num_nodes = self.graph_data.num_nodes

        # Generate masks
        indices = torch.randperm(num_nodes)
        train_size = int(train_ratio * num_nodes)
        val_size = (num_nodes - train_size) // 2

        train_mask = torch.zeros(num_nodes, dtype=torch.bool)
        val_mask = torch.zeros(num_nodes, dtype=torch.bool)
        test_mask = torch.zeros(num_nodes, dtype=torch.bool)

        train_mask[indices[:train_size]] = True
        val_mask[indices[train_size:train_size + val_size]] = True
        test_mask[indices[train_size + val_size:]] = True

        # Store masks
        if is_dgl:
            self.graph_data.ndata['train_mask'] = train_mask
            self.graph_data.ndata['val_mask'] = val_mask
            self.graph_data.ndata['test_mask'] = test_mask
        else:  # PyG
            self.graph_data.train_mask = train_mask
            self.graph_data.val_mask = val_mask
            self.graph_data.test_mask = test_mask

        print(f"Masks successfully generated and stored. (train_ratio={train_ratio})")

    def __repr__(self):
        return (f"Dataset(name={self.dataset_name}, api_type={self.api_type}, "
                f"#Nodes={self.num_nodes}, #Features={self.num_features}, "
                f"#Classes={self.num_classes})")


class Cora(Dataset):
    def __init__(self, api_type='dgl', path='./data'):
        super().__init__(api_type, path)

        if self.api_type == 'dgl':
            self.load_dgl_data()
        elif self.api_type == 'pyg':
            self.load_pyg_data()
        else:
            raise ValueError("Unsupported api_type.")

    def load_dgl_data(self):
        dataset = citation_graph.load_cora()
        data = dataset[0]
        self.graph_dataset = dataset
        self.graph_data = data
        self._load_meta_data()

    def load_pyg_data(self):
        dataset = Planetoid(root=self.path, name='Cora')
        data = dataset[0]
        self.graph_dataset = dataset
        self.graph_data = data
        self._load_meta_data()


class CiteSeer(Dataset):
    def __init__(self, api_type='dgl', path='./data'):
        super().__init__(api_type, path)

        if self.api_type == 'dgl':
            self.load_dgl_data()
        elif self.api_type == 'pyg':
            self.load_pyg_data()
        else:
            raise ValueError("Unsupported api_type.")

    def load_dgl_data(self):
        dataset = citation_graph.load_citeseer()
        data = dataset[0]
        self.graph_dataset = dataset
        self.graph_data = data
        self._load_meta_data()

    def load_pyg_data(self):
        dataset = Planetoid(root=self.path, name='Citeseer')
        data = dataset[0]
        self.graph_dataset = dataset
        self.graph_data = data
        self._load_meta_data()


class PubMed(Dataset):
    def __init__(self, api_type='dgl', path='./data'):
        super().__init__(api_type, path)

        if self.api_type == 'dgl':
            self.load_dgl_data()
        elif self.api_type == 'pyg':
            self.load_pyg_data()
        else:
            raise ValueError("Unsupported api_type.")

    def load_dgl_data(self):
        dataset = citation_graph.load_pubmed()
        data = dataset[0]
        self.graph_dataset = dataset
        self.graph_data = data
        self._load_meta_data()

    def load_pyg_data(self):
        dataset = Planetoid(root=self.path, name='PubMed')
        data = dataset[0]
        self.graph_dataset = dataset
        self.graph_data = data
        self._load_meta_data()


class Computers(Dataset):
    def __init__(self, api_type='dgl', path='./data'):
        super().__init__(api_type, path)

        if self.api_type == 'dgl':
            self.load_dgl_data()
        elif self.api_type == 'pyg':
            self.load_pyg_data()
        else:
            raise ValueError("Unsupported api_type.")

    def load_dgl_data(self):
        dataset = AmazonCoBuyComputerDataset(raw_dir=self.path)
        data = dataset[0]
        self.graph_dataset = dataset
        self.graph_data = data
        self._load_meta_data()

        self._generate_train_test_masks()

    def load_pyg_data(self):
        dataset = Amazon(root=self.path, name='Computers')
        data = dataset[0]
        self.graph_dataset = dataset
        self.graph_data = data
        self._load_meta_data()

        self._generate_train_test_masks()


class Photo(Dataset):
    def __init__(self, api_type='dgl', path='./data'):
        super().__init__(api_type, path)

        if self.api_type == 'dgl':
            self.load_dgl_data()
        elif self.api_type == 'pyg':
            self.load_pyg_data()
        else:
            raise ValueError("Unsupported api_type.")

    def load_dgl_data(self):
        dataset = AmazonCoBuyPhotoDataset(raw_dir=self.path)
        data = dataset[0]
        self.graph_dataset = dataset
        self.graph_data = data
        self._load_meta_data()

        self._generate_train_test_masks()

    def load_pyg_data(self):
        dataset = Amazon(root=self.path, name='Photo')
        data = dataset[0]
        self.graph_dataset = dataset
        self.graph_data = data
        self._load_meta_data()

        self._generate_train_test_masks()


class ENZYMES(Dataset):
    def __init__(self, api_type='dgl', path='./data'):
        super().__init__(api_type, path)

        if self.api_type == 'pyg':
            self.load_pyg_data()
        else:
            raise ValueError("Only pyg api_type is supported for ENZYMES.")

    def load_pyg_data(self):
        dataset = TUDataset(root=self.path, name='ENZYMES')
        data_list = [data for data in dataset]
        all_x = torch.cat([d.x for d in data_list], dim=0)
        mean, std = all_x.mean(0), all_x.std(0)
        for d in data_list:
            d.x = (d.x - mean) / (std + 1e-6)
        all_labels = np.array([int(d.y) for d in data_list])
        splitter = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
        train_idx, test_idx = next(splitter.split(np.zeros(len(all_labels)), all_labels))
        self.train_data = [data_list[i] for i in train_idx]
        self.test_data = [data_list[i] for i in test_idx]
