# Copyright 2023 parkminwoo, Apache 2.0 License

import re
from setuptools import find_packages
from setuptools import setup


def get_version():
    filename = "smilesfeature/__init__.py"
    with open(filename) as f:
        match = re.search(r"""^__version__ = ['"]([^'"]*)['"]""", f.read(), re.M)
    if not match:
        raise RuntimeError("{} doesn't contain __version__".format(filename))
    version = match.groups()[0]
    return version


def get_long_description():
    with open("README.md", encoding="UTF-8") as f:
        long_description = f.read()
        return long_description


version = get_version()

setup(
    name="smilesfeature",
    version="0.1.7",
    author="daniel park",
    author_email="parkminwoo1991@gmail.com",
    description="A Python package that automatically generates derived variables from a column with SMILES (Simplified Molecular-Input Line-Entry System).",
    long_description=get_long_description(),
    long_description_content_type="text/markdown",
    url="https://github.com/dsdanielpark/SMILES-feature",
    packages=find_packages(exclude=[]),
    python_requires=">=3.6",
    install_requires=[
        "numpy",
        "rdkit",
        "gensim",
        "mol2vec",
        "pandas",
        "scikit-learn",
        "IPython",
        "datamol",
        "molfeat"
    ],
    keywords="Python, SMILES, Cheminformatics, Molecular Informatics, Molecular Descriptor Generation, Chemical Data Analysis, Computational Chemistry",
    classifiers=[
        "Development Status :: 5 - Production/Stable",
        "Intended Audience :: Science/Research",
        "Natural Language :: English",
        "Programming Language :: Python",
        "Programming Language :: Python :: 3.6",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3.6",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
    entry_points={"console_scripts": ["bard_api=bard_api.cli:main"]},
)
