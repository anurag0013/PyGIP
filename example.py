from datasets import Cora
from models.defense import WatermarkByRandomGraph

# Example 1: Basic usage with default attack (ModelExtractionAttack0)
dataset = Cora()  # Load Cora dataset
defense = WatermarkByRandomGraph(dataset, attack_node_fraction=0.25)
results = defense.defend(attack_name="ModelExtractionAttack0")

# Example 2: Specifying attack at initialization
# dataset = Citeseer()  # Load Citeseer dataset
# defense = MyCustomDefense(dataset, attack_node_fraction=0.25, attack_name="ModelExtractionAttack2")
# results = defense.defend()

# Example 3: Specifying attack at defend() call
# dataset = Pubmed()  # Load Pubmed dataset
# defense = MyCustomDefense(dataset, attack_node_fraction=0.25)
# results = defense.defend(attack_name="ModelExtractionAttack3")

# Example 4: Using with different watermark settings
# dataset = Cora()
# defense = MyCustomDefense(
#     dataset, 
#     attack_node_fraction=0.3,  # Higher attack fraction
#     wm_node=100,               # More watermark nodes
#     pr=0.2,                    # Different feature probability
#     pg=0.1                     # Higher edge probability
# )
# results = defense.defend(attack_name="ModelExtractionAttack5")

# Print summary of all results
for i, result in enumerate([results], start=1):
    print(f"\nResult summary for Example {i}:")
    print(f"Watermark detection accuracy: {result['watermark_detection']:.4f}")
    if isinstance(result['target_attack_results'], dict) and 'success_rate' in result['target_attack_results']:
        print(f"Target model attack success: {result['target_attack_results']['success_rate']:.4f}")
    if isinstance(result['defense_attack_results'], dict) and 'success_rate' in result['defense_attack_results']:
        print(f"Defense model attack success: {result['defense_attack_results']['success_rate']:.4f}")