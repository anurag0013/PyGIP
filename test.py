from datasets import Cora
from models.attack import ModelExtractionAttack5 as MEA

from models.defense import BackdoorWM, SurviveWM

dataset = Cora()
mea = MEA(dataset, 0.1)
mea.attack()
