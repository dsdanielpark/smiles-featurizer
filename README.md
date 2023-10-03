Development Status :: 3 - Alpha


# SMILES featurizer

<p align="left">
<a href="https://github.com/dsdanielpark/SMILES-featurizer"><img alt="PyPI package" src="https://img.shields.io/badge/pypi-SMILES featurizer-black"></a>
<a href="https://github.com/psf/black"><img alt="Code style: black" src="https://img.shields.io/badge/code%20style-black-000000.svg"></a>
<a href="https://hits.seeyoufarm.com"><img src="https://hits.seeyoufarm.com/api/count/incr/badge.svg?url=https%3A%2F%2Fgithub.com%2Fdsdanielpark%2FSMILES-featurizer&count_bg=%23000000&title_bg=%23555555&icon=&icon_color=%23E7E7E7&title=hits&edge_flat=false"/></a>
<a href="https://github.com/dsdanielpark/SMILES-featurizer/stargazers"><img src="https://img.shields.io/github/stars/dsdanielpark/SMILES-featurizer?style=social"></a>
<a href="https://pypi.org/project/smilesfeaturizer/"><img alt="PyPI" src="https://img.shields.io/pypi/v/smilesfeaturizer"></a>
</p>

A Python package that automatically generates derived variables from a column with SMILES (Simplified Molecular-Input Line-Entry System)



## Install
```
$ pip install smilesfeaturizer
```
```
$ pip install git+https://github.com/dsdanielpark/SMILES-featurizer.git
```

## Usage 
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://drive.google.com/file/d/1BHTtOEvl577FyrQ5kLK-yJ9h9EDVUvGg/view?usp=sharing) 
- The dataset assumes the presence of SMILES strings in a column named `SMILES`.
### Feature generation
- Create fingerprint columns for SMILES representations based on various packages like RDKit, Mol2Vec, DataMol, MolFeat, sklearn, etc.
- Generate various derived variables to enable ML expansion.
- By default, it uses the 'simple' method, but if you provide the 'specific' argument, it creates various functions based on MACCS, FPVec, ECFP, and RDKit 2D descriptors, represented with suffixes like 'ecfp_'.
- You can vectorize the given SMILES string based on various derived variables, and they are created as columns with one-hot encoding for ML suitability.

    ```python
    from smilesfeaturizer import generate_smiles_feature

    df = generate_smiles_feature(df) # default method="simple"
    ```

<br>

### Create Dashboard 
- Through the dashboard, you can determine which compounds exhibit what prediction performance. 
- Researchers with domain knowledge can assess the prediction performance for specific molecules, identifying both good and poor performers, which can guide further modeling research.

    ```python
    from smilesfeaturizer import create_inline_dash_dashboard

    # Load your DataFrame and specify the columns
    true_col = 'pIC50'
    predicted_col = 'predicted_pIC50'

    # Create and run the Dash dashboard
    create_inline_dash_dashboard(df, true_col, predicted_col)
    ```

<br>

### Save reporting images
- Molecular images, basic information, and the prediction versus actual values are visually represented in bar graphs for easy viewing.
- If a storage path is provided, these images can be saved for later examination. Researchers can use this as a basis to collect additional features and determine the direction of modeling.
    ```python
    from smilesfeaturizer import smiles_insight_plot

    selected_metric = 'RMSE'  # Choose the error metric you want to display
    true_column = 'pIC50'  # Replace with your true column name
    predicted_column = 'predicted_pIC50'  # Replace with your predicted column name
    smiles_insight_plot(df[:1], true_column, predicted_column, selected_metric, 'output_folder', show=True)
    ```

<br>

## License
[Apache 2.0](https://opensource.org/license/apache-2-0/) <br>


## Bugs and Issues
Sincerely grateful for any reports on new features or bugs. Your valuable feedback on the code is highly appreciated.

## Contacts
- Core maintainer: [Daniel Park, South Korea](https://github.com/DSDanielPark) <br>
- E-mail: parkminwoo1991@gmail.com <br>

<br>

*Copyright (c) 2023 MinWoo Park, South Korea*<br>
