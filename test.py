from datasets import Cora
from models.attack.cega import CEGA

dataset = Cora()
cega = CEGA(dataset, attack_node_fraction=0.25)
cega.attack()
