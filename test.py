from datasets import Photo
from models.attack import DFEATypeIII as MEA

dataset = Photo(api_type='pyg')
print(dataset)
print(dataset.graph_dataset)
print(dataset.graph_data)
print(dataset.graph_data.train_mask.shape)


# mea = MEA(dataset, attack_node_fraction=0.1)
# mea.attack()
