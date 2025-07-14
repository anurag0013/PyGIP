from datasets import Cora
from models.defense import ATOM as MEA

dataset = Cora(api_type='pyg')
print(dataset)

mea = MEA(dataset, attack_node_fraction=0.1)
mea.defend()
