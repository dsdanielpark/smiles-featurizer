Development Status :: 3 - Alpha


# SMILES featurizer

## Feature generation
- Create fingerprint columns for SMILES representations based on various packages like RDKit, Mol2Vec, DataMol, MolFeat, sklearn, etc.

    ```python
    from smilesfeaturizer import generate_smiles_feature

    df = generate_smiles_feature(df) # default method="simple"
    ```


| Reaction Class                                           | Drug Substrates                               |
|----------------------------------------------------------|-----------------------------------------------|
| Cytochrome P450 - Dependent Oxidations (cont'd)          | Parathion                                     |
| Dechlorination                                           | Carbon tetrachloride                           |
| Cytochrome P450 Independent Oxidations: Flavin Monooxygenase (Ziegler's Enzyme) | Chlorpromazine, amitriptyline, benzphetamine, Desipramine, nortriptyline, Methimazole, Propylthiouracil |
| Amine Oxidations                                         | Phenylethylamine, Epinephrine                  |
| Dehydrogenations                                         | Ethanol                                       |
| Azo Reductions                                           | Prontosil, Tartarzine                          |
| Nitro Reductions                                         | Nitrobenzene, Chloramphenicol, Clorazepam, Dantrolene |
| Carbonyl Reductions                                      | Metyrapone, Methadone, Naloxone                |
| Ester Hydrolysis                                         | Procaine, Succinylcholine, Aspirin, Clofibrate, Methylphenidate |
| Amides Hydrolysis                                        | Procainamide, Lidocaine, Indomethacin          |
| Oxidation Cytochrome P450 - Dependent Oxidations: Aromatic Hydroxylations | Acetanilide, Propranolol, Phenobarbital, Phenytoin, Phenylbutazone, Amphetamine, Warfarin, 17a-ethinyl estradiol, Naphthalene, Benzpyrene |
| Aliphatic Hydroxylations                                 | Amobarbital, Pentobarbital, Secobarbital, Chlorpropamide, Ibuprofen, Meprobamate, Glutethimide, Phenylbutazone, Digitoxin |
| Epoxidation                                              | Aldrin                                        |
| Oxidative Dealkylation, N-Dealkylation                   | Morphine, Ethylmorphine, Benzphetamine, Aminopyrine, Caffeine, Theophylline |
| Oxidative Dealkylation, O-Dealkylation                   | Codeine, P-nitroanisole                        |
| Oxidative Dealkylation, S-Dealkylation                   | 6-Methylthiopurine, Methitural                |
| N-Oxidation, primary amines                              | Aniline, Chlorphentermine                      |
| N-Oxidation, secondary amines                            | 2-Acetylaminofluorene, Acetaminophen           |
| N-Oxidation, tertiary amines                             | Nicotine, Methaqualone                         |
| S-Oxidation                                              | Thioridazine, Cimetidine, Chlorpromazine       |
| Deamination                                              | Amphetamine, Diazepam                          |
| Desulfuration                                            | Thiopental                                    |


## Create dashboard 
- Through the dashboard, you can determine which compounds exhibit what prediction performance. 

    ```python
    from smilesfeaturizer import create_inline_dash_dashboard

    # Load your DataFrame and specify the columns
    true_col = 'pIC50'
    predicted_col = 'predicted_pIC50'

    # Create and run the Dash dashboard
    create_inline_dash_dashboard(df, true_col, predicted_col)
    ```

## Save reporting images
- Molecular images, basic information, and the prediction versus actual values are visually represented in bar graphs for easy viewing.
    ```python
    from smilesfeaturizer import smiles_insight_plot

    selected_metric = 'RMSE'  # Choose the error metric you want to display
    true_col = 'pIC50'  # Replace with your true column name
    predicted_col = 'predicted_pIC50'  # Replace with your predicted column name
    smiles_insight_plot(df[:1], true_col, predicted_col, selected_metric, 'output_folder', show=True)
    ```

    ![](./output_folder/1.jpg)

<br>
