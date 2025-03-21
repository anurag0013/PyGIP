from datasets import *
from models.attack import AdversarialModelExtraction

dataset = Cora()
mea = AdversarialModelExtraction(dataset, 0.25)
mea.attack()
