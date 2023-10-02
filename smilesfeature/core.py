import os
import re
import numpy as np
import pandas as pd
import joblib
from rdkit import Chem
from sklearn.cluster import DBSCAN
from gensim.models import Word2Vec
import pkg_resources
from smilesfeature.processor.mol2vec_processor import sentences2vec, mol2vec_feature
from smilesfeature.processor.smiles_processor import (
    add_molecule_from_smiles,
    smiles_to_fp,
    generate_3D_coordinates,
    smiles_to_image_array,
    add_all_descriptors,
    find_reactive_sites,
    count_reactive_sites,
    count_reaction_fragments,
    add_reactive_groups,
    generate_descriptor_functions,
    apply_pca_to_dataframe,
    add_chem_properties,
    expand_reaction_sites,
    generate_chemical_properties,
    interpolate_missing_values,
    perform_pca_on_mol2vec,
    extract_extra_features,
)
from smilesfeature.constant import REACTION_CLASSES_TO_SMILES_FRAGMENTS

data_path = pkg_resources.resource_filename('smilesfeature.data', 'model_300dim.pkl')


def feature_generate(df, smiles_col="SMILES", method="simple"):
    """
    Generate derived variables from SMILES in a DataFrame.

    Args:
        df (pd.DataFrame): Input DataFrame containing a smiles_col column.
        method (str, optional): Method for feature generation. Defaults to "simple".

    Returns:
        pd.DataFrame: DataFrame with derived features added.

    Example:
        >>> import pandas as pd
        >>> data = {"SMILES": ['CCO', 'C1=CC=CC=C1', 'CCC']}
        >>> df = pd.DataFrame(data)
        >>> result_df = generate(df, method='more')
        >>> print(result_df.head())

    Note:
        This function performs a series of preprocessing, feature engineering,
        and chemical property calculation steps to generate derived features
        from the smiles_col column in the input DataFrame. default is "SMILES".
    """
    mol2vec_model = Word2Vec.load(data_path)
    # Preprocessing steps
    df = interpolate_missing_values(df)
    df = add_molecule_from_smiles(df)
    df["Mol"] = df[smiles_col].apply(Chem.MolFromSmiles)
    df = generate_chemical_properties(df)

    # Feature Engineering steps
    df["fingerprint"] = df[smiles_col].apply(smiles_to_fp)
    df[["3D_Coordinates", "Mol2Vec", "image_array"]] = df[smiles_col].apply(
        lambda x: pd.Series(
            {
                "3D_Coordinates": generate_3D_coordinates(x),
                "Mol2Vec": mol2vec_feature(x, mol2vec_model).reshape(-1),
                "image_array": smiles_to_image_array(x),
            }
        )
    )
    df = apply_pca_to_dataframe(df, '3D_Coordinates')
    df['Mol2Vec'] = df['Mol2Vec'].apply(lambda x: x.reshape(-1))
    df = perform_pca_on_mol2vec(df, n_components=3)
    df["Mol2Vec_mean"] = df["Mol2Vec"].apply(np.mean)
    df["Mol2Vec_std"] = df["Mol2Vec"].apply(np.std)

    # Chemical Property steps
    df = add_reactive_groups(df)
    df = add_all_descriptors(df)
    df['Reactive_Sites'] = df[smiles_col].apply(lambda x: find_reactive_sites(Chem.MolFromSmiles(x)))
    df["Reactive_Sites_Count"] = df["Molecule"].apply(count_reactive_sites)
    for site in [
        "Nitro_Reduction",
        "Carbonyl_Reduction",
        "Dehydration",
        "Amides",
        "Oxidations",
        "Epoxidation",
        "Oxidative_Dealkylation",
    ]:
        df[site] = df["Reactive_Sites_Count"].apply(lambda x: x.get(site, 0))

    # Clustering
    mol2vec_matrix = np.array(df["Mol2Vec"].tolist())
    dbscan = DBSCAN(eps=1, min_samples=2)
    df["Cluster"] = dbscan.fit_predict(mol2vec_matrix)

    # Reaction steps
    df = count_reaction_fragments(df, REACTION_CLASSES_TO_SMILES_FRAGMENTS)
    df = add_chem_properties(df)

    # Clean column names
    df.columns = [re.sub("[^0-9a-zA-Z]+", "_", col) for col in df.columns]

    # Reaction Site expansion
    df = expand_reaction_sites(df)

    # Optional additional features
    if method == "more":
        df = extract_extra_features(df)
        df.drop(columns=["clogp", "mw", "tpsa"], inplace=True)

    return df
