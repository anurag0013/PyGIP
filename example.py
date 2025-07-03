from datasets import Cora

dataset = Cora()

# >>>>>>>>>> test MEA

# mea = ModelExtractionAttack0(dataset, 0.25)
# mea.attack()


# >>>>>>>>>> test ImperceptibleWM
# from models.defense.OwnerWatermarkingDefense import OwnerWatermarkingDefense
# from utils.dglTopyg import dgl_to_pyg_data
#
# pyg_data = dgl_to_pyg_data(dataset.graph)
#
# defense = OwnerWatermarkingDefense(dataset)
# metrics = defense.defend(pyg_data)
#
# print(metrics)


# >>>>>>>>>> test ImperceptibleWM2
# from models.defense.ImperceptibleOwnerUniqueWatermark import WatermarkByBilevelOptimization
#
# defense = WatermarkByBilevelOptimization(dataset)
# defense.defend()


# >>>>>>>>>> test SurviveWM2
from models.defense.SurviveWM2 import OptimizedWatermarkDefense
from datasets import ENZYMES

dataset = ENZYMES()
defense = OptimizedWatermarkDefense(dataset, 0.25)
defense.defend()
