from datasets import CiteSeer
from models.attack import CEGA as MEA

dataset = CiteSeer(api_type='dgl')
print(dataset)

mea = MEA(dataset, attack_node_fraction=0.1)
mea.attack(EVAL_EPOCH=2, TGT_EPOCH=2, WARMUP_EPOCH=1)
