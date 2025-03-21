from datasets import Cora
from models.attack import ModelExtractionAttack0

dataset = Cora()
mea = ModelExtractionAttack0(dataset, 0.25)
mea.attack()
