

from .data_free.base import DataFreeBaseAttack
from .data_free.type_i import TypeIAttack
from .data_free.type_ii import TypeIIAttack
from .data_free.type_iii import TypeIIIAttack

class DataFreeAttack(DataFreeBaseAttack):
    """Data-free model extraction attack for GNNs.
    
    This class provides a unified interface for different types of
    data-free attacks on graph neural networks.
    """
    
    def __init__(self, attack_type=1, **kwargs):
        """Initialize data-free attack.
        
        Args:
            attack_type (int): Type of attack (1, 2, or 3)
            **kwargs: Additional arguments for specific attack types
        """
        super().__init__(**kwargs)
        self.attack_type = attack_type
        self.attack_impl = None
        
    def initialize_models(self):
        """Initialize attack implementation based on type."""
        if self.attack_type == 1:
            self.attack_impl = TypeIAttack(
                self.generator_config,
                self.surrogate_config
            )
        elif self.attack_type == 2:
            self.attack_impl = TypeIIAttack(
                self.generator_config,
                self.surrogate_config
            )
        else:
            self.attack_impl = TypeIIIAttack(
                self.generator_config,
                self.surrogate_config
            )
