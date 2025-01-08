from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="pygip",
    version="0.1.0",
    author="Your Name",
    author_email="your.email@example.com",
    description="A Python package for Graph Intellectual Property Protection",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/pygip",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.7",
    install_requires=[
        "torch>=1.7.0",
        "dgl>=0.6.0",
        "torch-geometric>=2.0.0",
        "numpy>=1.19.0",
        "scipy>=1.6.0",
        "networkx>=2.5",
        "scikit-learn>=0.24.0",
        "tqdm>=4.50.0",
    ],
    extras_require={
        "dev": [
            "pytest>=6.0",
            "pytest-cov>=2.0",
            "flake8>=3.9.0",
            "black>=21.5b2",
            "isort>=5.8.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "pygip=pygip.cli:main",
        ],
    },
    package_data={
        "pygip": [
            "data/*",
            "models/*",
        ],
    },
    include_package_data=True,
)