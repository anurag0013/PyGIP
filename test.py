from datasets import Cora, PubMed
from models.attack import CEGA as MEA

dataset = PubMed(api_type='pyg')
print(dataset)

mea = MEA(dataset, attack_node_fraction=0.1)
mea.attack()
