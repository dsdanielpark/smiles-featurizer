Development Status :: 3 - Alpha


# SMILES featurizer

<p align="left">
<a href="https://github.com/dsdanielpark/smiles-featurizer"><img alt="PyPI package" src="https://img.shields.io/badge/pypi-SMILES featurizer-black"></a>
<a href="https://github.com/psf/black"><img alt="Code style: black" src="https://img.shields.io/badge/code%20style-black-000000.svg"></a>
<a href="https://hits.seeyoufarm.com"><img src="https://hits.seeyoufarm.com/api/count/incr/badge.svg?url=https%3A%2F%2Fgithub.com%2Fdsdanielpark%2FSMILES-featurizer&count_bg=%23000000&title_bg=%23555555&icon=&icon_color=%23E7E7E7&title=hits&edge_flat=false"/></a>
<a href="https://pypi.org/project/smilesfeaturizer/"><img alt="PyPI" src="https://img.shields.io/pypi/v/smilesfeaturizer"></a>
</p>

A Python package that automatically generates derived feature variables from a column with SMILES (Simplified Molecular-Input Line-Entry System)

![](./assets/smilesfeaturizer.gif)


The python package, SMILES Featurizer helps quickly and painlessly explore the baseline and key features for many projects that use SMILES strings. In some cases, there may be duplicate columns among the columns created by the specific method. Clean and use some of the columns in the generated data frame. *I intentionally did not encapsulate it highly as a class, and I maintain it in the form of functions. This is because it is based on the processing of a single data frame and because the service is highly likely to be modified.*

<br>

## Install
```
$ pip install smilesfeaturizer
```
```
$ pip install git+https://github.com/dsdanielpark/SMILES-featurizer.git
```
<br>

## Usage [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://drive.google.com/file/d/1BHTtOEvl577FyrQ5kLK-yJ9h9EDVUvGg/view?usp=sharing) 
The dataset assumes the presence of SMILES strings in a column named SMILES. See [tutorial notebook](https://github.com/dsdanielpark/SMILES-featurizer/blob/main/tutorial.ipynb).
### *Feature generation*
- Create fingerprint columns for SMILES representations based on various packages [RDKit](https://www.rdkit.org/), [Mol2Vec](https://github.com/samoturk/mol2vec), [DataMol](https://github.com/datamolorg/datamol), [MolFeat](https://github.com/cplassier/molfeat), [Scikit-Learn](https://scikit-learn.org/stable/).

    ```python
    from smilesfeaturizer import generate_smiles_feature

    df = generate_smiles_feature(df) # default method="simple"

    df = generate_smiles_feature(df, method="specific") 
    ```

### *Create dashboard* 
- Through the dashboard, you can determine which compounds exhibit what prediction performance. 

    ```python
    from smilesfeaturizer import create_inline_dash_dashboard

    # Load your DataFrame and specify the columns
    true_col = 'pIC50'
    predicted_col = 'predicted_pIC50'

    # Create and run the Dash dashboard
    create_inline_dash_dashboard(df, true_col, predicted_col)
    ```

### *Save reporting images*
- Molecular images, basic information, and the prediction versus actual values are visually represented in bar graphs for easy viewing.
    ```python
    from smilesfeaturizer import smiles_insight_plot

    selected_metric = 'RMSE'  # Choose the error metric you want to display
    true_col = 'pIC50'  # Replace with your true column name
    predicted_col = 'predicted_pIC50'  # Replace with your predicted column name
    smiles_insight_plot(df[:1], true_col, predicted_col, selected_metric, 'output_folder', show=True)
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
