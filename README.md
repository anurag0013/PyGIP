# PyGIP

PyGIP is a Python library designed for experimenting with graph-based model extraction attacks and defenses. It provides
a modular framework to implement and test attack and defense strategies on graph datasets.

## Installation

To get started with PyGIP, set up your environment by installing the required dependencies:

```bash
pip install -r reqs.txt
```

Ensure you have Python installed (version 3.8 or higher recommended) along with the necessary libraries listed
in `reqs.txt`.

## Quick Start

Here’s a simple example to launch a model extraction attack using PyGIP:

```python
from datasets import Cora
from models.attack import ModelExtractionAttack0

# Load the Cora dataset
dataset = Cora()

# Initialize the attack with a sampling ratio of 0.25
mea = ModelExtractionAttack0(dataset, 0.25)

# Execute the attack
mea.attack()
```

This code loads the Cora dataset, initializes a basic model extraction attack (`ModelExtractionAttack0`), and runs the
attack with a specified sampling ratio.
Here’s an expanded and detailed version of the "Contribute to Code" section for your README.md, incorporating the
specifics of `BaseAttack` and `Dataset` you provided. This version is thorough, clear, and tailored for contributors:

## Contribute to Code

PyGIP is built to be modular and extensible, allowing contributors to implement their own attack and defense strategies.
Below, we detail how to extend the framework by implementing custom attack and defense classes, with a focus on how to
leverage the provided dataset structure.

### Implementing an Attack

To create a custom attack, you need to extend the abstract base class `BaseAttack`. Here’s the structure
of `BaseAttack`:

```python
from abc import ABC
from datasets import Dataset


class BaseAttack(ABC):
    def __init__(self, dataset: Dataset, attack_node_fraction: float, model_path: str = None):
        """Base class for all attack implementations."""
        self.dataset = dataset
        self.graph = dataset.graph  # Access the DGL-based graph directly
        # Additional initialization can go here

    def attack(self):
        """Abstract method to implement attack logic."""
        pass
```

To implement your own attack:

1. **Inherit from `BaseAttack`**:
   Create a new class that inherits from `BaseAttack`. You’ll need to provide the required parameters in the
   constructor:
    - `dataset`: An instance of the `Dataset` class (see below for details).
    - `attack_node_fraction`: A float between 0 and 1 representing the fraction of nodes to attack.
    - `model_path` (optional): A string specifying the path to a pre-trained model (defaults to `None`).

2. **Implement the `attack()` Method**:
   Override the abstract `attack()` method with your attack logic. For example:

   ```python
   class MyCustomAttack(BaseAttack):
       def __init__(self, dataset: Dataset, attack_node_fraction: float, model_path: str = None):
           super().__init__(dataset, attack_node_fraction, model_path)
           # Additional initialization if needed

       def attack(self):
           # Example: Access the graph and perform an attack
           print(f"Attacking {self.attack_node_fraction * 100}% of nodes")
           num_nodes = self.graph.num_nodes()
           print(f"Graph has {num_nodes} nodes")
           # Add your attack logic here
   ```

3. **Accessing the Dataset**:
    - The `dataset` passed to `BaseAttack` is an instance of the `Dataset` class (see below).
    - Use `self.graph` to directly access the DGL-based graph data. This is pre-populated by the `Dataset` class and
      provides a unified interface to the graph structure.
    - You can also access other dataset attributes
      like `self.dataset.features`, `self.dataset.labels`, `self.dataset.train_mask`, etc., if needed.

4. **Adding Helper Functions**:
   Feel free to add helper methods within your class to modularize your attack logic. For example:

   ```python
   class MyCustomAttack(BaseAttack):
       def __init__(self, dataset: Dataset, attack_node_fraction: float, model_path: str = None):
           super().__init__(dataset, attack_node_fraction, model_path)

       def _select_nodes(self):
           # Helper function to select nodes for attack
           num_nodes = self.graph.num_nodes()
           attack_size = int(num_nodes * self.attack_node_fraction)
           return range(attack_size)  # Example selection

       def attack(self):
           target_nodes = self._select_nodes()
           print(f"Attacking {len(target_nodes)} nodes")
           # Attack logic here
   ```

### Implementing a Defense

To create a custom defense, extend the `BaseDefense` class:

```python
from models.defense import BaseDefense


class MyCustomDefense(BaseDefense):
    def defend(self):
        # Add your defense logic here
        pass
```

A typical defense workflow might look like this:

1. Train a target model using the dataset.
2. Perform an attack on the target model (e.g., using an attack class) and evaluate its performance.
3. Train a defense model to mitigate the attack.
4. Test the defense model against the same attack and report the performance.

Example:

```python
class MyCustomDefense(BaseDefense):
    def defend(self):
        # Step 1: Train target model
        target_model = self._train_target_model()
        # Step 2: Attack target model
        attack = MyCustomAttack(self.dataset, attack_node_fraction=0.3)
        attack.attack(target_model)
        # Step 3: Train defense model
        defense_model = self._train_defense_model()
        # Step 4: Test defense against attack
        attack = MyCustomAttack(self.dataset, attack_node_fraction=0.3)
        attack.attack(defense_model)
        # Print performance metrics

    def _train_target_model(self):
        # Helper function for training target model
        pass

    def _train_defense_model(self):
        # Helper function for training defense model
        pass
```

### Understanding the Dataset Class

The `Dataset` class standardizes the data format across PyGIP. Here’s its structure:

```python
class Dataset(object):
    def __init__(self, api_type='dgl', path='./downloads/'):
        self.api_type = api_type  # Set to 'dgl' for DGL-based graphs
        self.path = path  # Directory for dataset storage
        self.dataset_name = ""  # Name of the dataset (e.g., "Cora")

        # Graph properties
        self.node_number = 0  # Number of nodes
        self.feature_number = 0  # Number of features per node
        self.label_number = 0  # Number of label classes

        # Core data
        self.graph = None  # DGL graph object
        self.features = None  # Node features
        self.labels = None  # Node labels

        # Data splits
        self.train_mask = None  # Boolean mask for training nodes
        self.val_mask = None  # Boolean mask for validation nodes
        self.test_mask = None  # Boolean mask for test nodes
```

- **Key Insight for Contributors**: You don’t need to worry about loading or formatting the dataset manually. Simply
  use `self.graph` in your attack or defense class to access the DGL-based graph object. This ensures consistency across
  implementations.
- Additional attributes like `self.dataset.features` (node features), `self.dataset.labels` (node labels),
  and `self.dataset.train_mask` (training split) are also available if your logic requires them.

### Submit Guideline

please submit pull request
todo...


### Miscellaneous Tips

- **Reference Implementation**: The `ModelExtractionAttack0` class is a fully implemented attack example. Study it for
  inspiration or as a template.
- **Flexibility**: Add as many helper functions as needed within your class to keep your code clean and modular.
- **Dataset Access**: Always use `self.graph` for graph operations to maintain compatibility with the framework’s
  DGL-based structure.
- **Backbone Models**: We provide several basic backbone models like `GCN, GraphSAGE`. You can use or add more
  at `from models.nn import GraphSAGE`.

By following these guidelines, you can seamlessly integrate your custom attack or defense strategies into PyGIP. Happy
coding!

## License

MIT License

## Contact

For questions or contributions, please contact blshen@fsu.edu.
