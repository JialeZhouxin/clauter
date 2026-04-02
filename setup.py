"""Setup script for clauter."""
from setuptools import setup, find_packages

setup(
    name="clauter",
    version="0.1.0",
    description="PyTorch implementation of DEC and IDEC deep clustering algorithms",
    author="Your Name",
    author_email="your.email@example.com",
    url="https://github.com/yourusername/clauter",
    packages=find_packages(include=["deepclust_base*", "scripts*"]),
    python_requires=">=3.8",
    install_requires=[
        "torch>=2.0.0",
        "numpy>=1.24.0",
        "scipy>=1.10.0",
        "scikit-learn>=1.3.0",
        "pyyaml>=6.0",
        "munkres>=3.0.0",
    ],
    extras_require={
        "dev": ["pytest>=7.0"],
    },
    entry_points={
        "console_scripts": [
            "train-dec=scripts.train_dec_idec:main",
        ],
    },
    classifiers=[
        "Development Status :: 3 - Alpha",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
    license="MIT",
)
