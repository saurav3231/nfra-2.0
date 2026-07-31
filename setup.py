"""
setup.py for NFRA 2.0

This file is provided for compatibility with older tools.
Modern installation should use: pip install -e .

Created by Saurav Bhandari
"""

from setuptools import setup, find_packages
import os

# Read long description from README
here = os.path.abspath(os.path.dirname(__file__))
with open(os.path.join(here, "README.md"), encoding="utf-8") as f:
    long_description = f.read()

setup(
    name="nfra",
    version="3.1.0",
    description="NeuroFractal Resonance Architecture - Brain-inspired efficient neural networks",
    long_description=long_description,
    long_description_content_type="text/markdown",
    author="SAURAV BHANDARI",
    author_email="bhandarisaurav15@gmail.com",
    url="https://github.com/saurav3231/nfra-2.0",
    project_urls={
        "Documentation": "https://github.com/saurav3231/nfra-2.0#readme",
        "Source": "https://github.com/saurav3231/nfra-2.0",
        "Tracker": "https://github.com/saurav3231/nfra-2.0/issues",
    },
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    python_requires=">=3.9",
    install_requires=[
        "torch>=2.0.0",
        "numpy>=1.23.0",
        "pyyaml>=6.0",
        "tqdm>=4.65.0",
    ],
    extras_require={
        "dev": [
            "pytest>=7.0",
            "black>=23.0",
            "ruff>=0.1.0",
            "datasets>=2.14.0",
            "transformers>=4.30.0",
        ],
        "all": [
            "datasets>=2.14.0",
            "transformers>=4.30.0",
            "accelerate>=0.20.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "nfra-train=scripts.train_nfra:main",
            "nfra-eval=scripts.evaluate_nfra:main",
        ],
    },
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Scientific/Engineering :: Information Analysis",
    ],
    keywords=[
        "neural-networks",
        "efficient-ai",
        "edge-ai",
        "fractal",
        "predictive-coding",
        "low-power",
        "brain-inspired",
    ],
    license="MIT",
    include_package_data=True,
    zip_safe=False,
)