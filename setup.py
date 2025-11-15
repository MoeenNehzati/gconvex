from setuptools import setup, find_packages
import io
import os

here = os.path.abspath(os.path.dirname(__file__))
long_description = """
gconvex: Finitely convex parametrization of generalized convex functions
and experimental baselines for Monge map / optimal transport comparisons.
"""
try:
    with io.open(os.path.join(here, "README.md"), encoding="utf-8") as f:
        long_description = f.read()
except Exception:
    pass

setup(
    name="gconvex",
    version="0.1.0",
    description="Finitely convex parametrization and OT baselines",
    long_description=long_description,
    long_description_content_type="text/markdown",
    author="Moeen Nehzati",
    url="https://github.com/MoeenNehzati/gconvex",
    packages=find_packages(exclude=("tests", "docs")),
    include_package_data=True,
    python_requires=">=3.8",
    install_requires=[
        # Core numeric / ML
        "torch>=1.9.0,<2.0",
        "torchvision>=0.10.0,<0.15",
        "numpy>=1.19.0,<2.0",
        # Visualization & data
        "matplotlib>=3.3.0,<4.0",
        "pandas>=1.1.0,<2.0",
        "Pillow>=8.0.0,<10.0",
        # Utilities
        "tqdm>=4.50.0",
        "h5py>=3.0.0",
        "rich>=10.0.0",
        "ipython>=7.0.0",
    ],
    extras_require={
        "dev": ["pytest", "flake8"],
        "wandb": ["wandb"],
    },
    license="Apache-2.0",
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: Apache Software License",
        "Operating System :: OS Independent",
    ],
)
