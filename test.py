from datasets import Cora
from models.attack import ModelExtractionAttack0

from models.defense.atom.ATOM import ATOM

dataset = Cora(api_type='torch_geometric')
print(dataset.node_number)
atom = ATOM(dataset, attack_node_fraction=0.25)
atom.defend()
