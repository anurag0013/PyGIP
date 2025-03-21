# PyGIP

PyGIP is a Python library designed for experimenting with graph-based model extraction attacks and defenses. It provides a modular framework to implement and test attack and defense strategies on graph datasets.

## Installation

To get started with PyGIP, set up your environment by installing the required dependencies:

```bash
pip install -r reqs.txt
```

Ensure you have Python installed (version 3.8 or higher recommended) along with the necessary libraries listed in `reqs.txt`.

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

This code loads the Cora dataset, initializes a basic model extraction attack (`ModelExtractionAttack0`), and runs the attack with a specified sampling ratio.

## Contribute to Code

PyGIP is designed to be extensible. You can contribute by implementing your own attack or defense strategies.

### Implementing an Attack

To create a custom attack, implement the `BaseAttack` class:

```python
from models.attack import BaseAttack

class MyCustomAttack(BaseAttack):
    def attack(self):
        # Add your attack logic here
        pass
```

- Inherit from `BaseAttack`.
- Define your attack logic in the `attack()` method.
- Use `self.graph` to access the DGL-based graph data provided by the dataset.

### Implementing a Defense

To create a custom defense, implement the `BaseDefense` class:

```python
from models.defense import BaseDefense

class MyCustomDefense(BaseDefense):
    def defend(self):
        # Add your defense logic here
        pass
```

A typical `defend()` workflow should include:
1. Train a target model.
2. Perform an attack on the target model and print the attack performance.
3. Train a defense model.
4. Perform an attack on the defense model and print the defense performance.

### Miscellaneous Notes

- **Reference Implementation**: Check out `ModelExtractionAttack0` for a fully implemented attack class as an example.
- **Dataset Access**: All datasets are standardized. Use `self.graph` to access the DGL-based graph data in your attack or defense class.
- **Helper Functions**: Feel free to add helper functions within your custom class to support your logic.

## License

MIT License

## Contact

For questions or contributions, please contact blshen@fsu.edu.
