from datasets import Cora
from models.attack.dfea import *

dataset = Cora()  # Load Cora dataset
attack = DFEATypeI(dataset=dataset, attack_node_fraction=0.25)

accuracy = attack.attack()
print(f"DFEATypeI completed. Surrogate-Victim agreement accuracy: {accuracy:.4f}")