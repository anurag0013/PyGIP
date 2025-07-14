from datasets import CiteSeer
from models.defense import RandomWM as MEA

dataset = CiteSeer(api_type='dgl')
print(dataset)

mea = MEA(dataset, attack_node_fraction=0.1)
mea.defend()
