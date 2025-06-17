from .base import BaseDefense
from .WatermarkDefense import (
    WatermarkByRandomGraph,
    BaseDefense
)

from .BackdoorWM import BackdoorWM
from .SurviveWM import SurviveWM

__all__ = [
    'BaseDefense',
    'WatermarkByRandomGraph',
    'BackdoorWM',
    'SurviveWM',
]
