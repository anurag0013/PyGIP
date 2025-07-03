from datasets import Cora
from models.attack import ModelExtractionAttack0

from models.defense import BackdoorWM, SurviveWM

dataset = Cora()
mea = BackdoorWM(dataset, 0.25)
mea.defend()
