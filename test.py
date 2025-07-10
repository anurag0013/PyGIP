from datasets import Cora
from models.attack import AdvMEA as MEA

from models.defense import ImperceptibleWM as DFS

dataset = Cora()
mea = MEA(dataset, 0.1)
mea.attack()
