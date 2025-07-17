from datasets import Cora, PubMed
from models.attack import CEGA as MEA
from models.defense import ATOM as MEAD

dataset = Cora(api_type='pyg')
print(dataset)

mea = MEAD(dataset, attack_node_fraction=0.1)
mea.defend()
