from datasets import Cora
from models.attack import ModelExtractionAttack5 as MEA

from models.defense import ImperceptibleWM as DFS

dataset = Cora()
dfs = DFS(dataset, 0.1)
dfs.defend()
