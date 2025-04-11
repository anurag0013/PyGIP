from .models.attack import BaseAttack
from .models.attack.adversarial import AdversarialModelExtraction
from .models.attack.mea import (
    ModelExtractionAttack0,
    ModelExtractionAttack1,
    ModelExtractionAttack2,
    ModelExtractionAttack3,
    ModelExtractionAttack4,
    ModelExtractionAttack5
)
from .models.defense import BaseDefense
from .models.defense.WatermarkDefense import WatermarkByRandomGraph
from .models.nn import GCN, GraphSAGE, ShadowNet, AttackNet
from .utils.metrics import GraphNeuralNetworkMetric

__version__ = "0.1.0"

__all__ = [
    # Base classes
    'BaseAttack',
    'BaseDefense',

    # Attack implementations
    'ModelExtractionAttack0',
    'ModelExtractionAttack1',
    'ModelExtractionAttack2',
    'ModelExtractionAttack3',
    'ModelExtractionAttack4',
    'ModelExtractionAttack5',
    'AdversarialModelExtraction',

    # Defense implementations
    'Watermark_sage',

    # Utility classes
    'GraphNeuralNetworkMetric',
    'GCN',
    'GraphSAGE',
    'ShadowNet',
    'AttackNet'
]
