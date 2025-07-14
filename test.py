from datasets import CiteSeer
from models.defense import ATOM as MEAD

dataset = CiteSeer(api_type='pyg')
print(dataset)

mead = MEAD(dataset, attack_node_fraction=0.1)
mead.defend()
